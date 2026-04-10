import torch
import torch.nn as nn

class CMERTClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import AutoModel
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)
        self.cmert = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)

        for param in self.cmert.parameters():
            param.requires_grad = False
        # for param in self.cmert.feature_extractor.parameters():
        #     param.requires_grad = True
        for param in self.cmert.encoder.layers[-1].parameters():
            param.requires_grad = True

        self.embed_dim = config.embed_dims
        self.dropout = config.dropout
        self.num_classes = config.num_classes
        self.prob = nn.Sequential(nn.Linear(768, self.embed_dim),
                                  nn.ReLU(),
                                  nn.Dropout(self.dropout) if self.dropout else nn.Identity(),
                                  nn.Linear(self.embed_dim, self.num_classes))

    def _embed_from_cmert(self, x):
        out = self.processor(x.detach().cpu().numpy(), sampling_rate=24000, return_tensors='pt')
        out = {k:v.to(x.device) for k,v in out.items()}
        # with torch.no_grad():
        #     out = self.cmert(**out, output_hidden_states=False)

        out = self.cmert(**out, output_hidden_states=False)
        # input_values = self.processor(x, sampling_rate=24000, return_tensors="pt").input_values.squeeze(0).to(x.device)

        # with torch.no_grad():
        #     out = self.cmert.feature_extractor(input_values).transpose(1, 2)
        #     out = self.cmert.feature_projection(out)
        #     out = self.cmert.encoder(out, output_hidden_states=False)

        out = out.last_hidden_state # b,t,c

        return out


    def forward(self, x):
        out = self._embed_from_cmert(x) # b,t,c
        out = self.prob(out) # b,t,l

        return out



import torch.nn as nn
from models.modules import Conv1DBlock
from models.model_utils import calc_conv_param

class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims
    def forward(self, x):
        return x.permute(*self.dims)

class CMERTConv1DEnc(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes

        self.params = calc_conv_param(config)
        self.prob = nn.Sequential()
        self.build()

        from transformers import AutoModel
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)
        self.cmert = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)

        for param in self.cmert.parameters():
            param.requires_grad = False


    def build(self):
        for idx, param in enumerate(self.params):
            self.prob.add_module(f'conv_{idx}', Conv1DBlock(param['input_channel'], param['output_channel'],
                                                           kernel_size=self.kernel_size, padding='same', dilation=self.dilation))
            if self.config.cnn_dropout: self.prob.add_module(f'dropout_{idx}', nn.Dropout1d(self.config.cnn_dropout))
            if self.config.pool_size: self.prob.add_module(f'pool_{idx}', nn.MaxPool1d(param['max_pool']))
        self.prob.add_module('permute', Permute(0,2,1))
        self.prob.add_module('fc', nn.Linear(self.params[-1]['output_channel'], self.num_classes))

    def _embed_from_cmert(self, x):
        input_values = self.processor(x, sampling_rate=24000, return_tensors="pt").input_values.squeeze(0).to(x.device)

        with torch.no_grad():
            out = self.cmert.feature_extractor(input_values).transpose(1, 2)
            out = self.cmert.feature_projection(out)
            out = self.cmert.encoder(out, output_hidden_states=False)

        out = out.last_hidden_state.permute(0,2,1) # b,c,t

        return out

    def forward(self, x):
        x = self._embed_from_cmert(x) # b,c,t
        x = self.prob(x) # b,h,t->b,t,h->b,t,l
        return x


class CMERTConv1DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes

        self.dropout = config.dropout
        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru

        self.params = calc_conv_param(config)
        self.enc = nn.Sequential()
        self.build()

        self.gru = nn.GRU(input_size=self.params[-1]['output_channel'],
                          hidden_size=self.hidden_dim,
                          num_layers=self.num_gru,
                          batch_first=True,
                          dropout=self.dropout,
                          bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim*2,
                            self.num_classes)

        from transformers import AutoModel
        from transformers import Wav2Vec2FeatureExtractor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)
        self.cmert = AutoModel.from_pretrained("ntua-slp/CultureMERT-95M", trust_remote_code=True)

        for param in self.cmert.parameters():
            param.requires_grad = False


    def build(self):
        for idx, param in enumerate(self.params):
            self.enc.add_module(f'conv_{idx}', Conv1DBlock(param['input_channel'], param['output_channel'],
                                                           kernel_size=self.kernel_size, padding='same', dilation=self.dilation))
            if self.config.cnn_dropout: self.enc.add_module(f'dropout_{idx}', nn.Dropout1d(self.config.cnn_dropout))
            if self.config.pool_size: self.enc.add_module(f'pool_{idx}', nn.MaxPool1d(param['max_pool']))

    def _embed_from_cmert(self, x):
        input_values = self.processor(x, sampling_rate=24000, return_tensors="pt").input_values.squeeze(0).to(x.device)

        with torch.no_grad():
            out = self.cmert.feature_extractor(input_values).transpose(1, 2)
            out = self.cmert.feature_projection(out)
            out = self.cmert.encoder(out, output_hidden_states=False)

        out = out.last_hidden_state

        return out

    def forward(self, x):
        x = self._embed_from_cmert(x).permute(0,2,1) # b,t,c -> b,c,t
        x = self.enc(x).permute(0,2,1) # b,h,t -> b,t,h
        x,_ = self.gru(x) # b,t,2h
        x = self.fc(x) # b,t,l
        return x