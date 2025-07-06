from models.modules import Conv2DBlock
from models.model_utils import calc_conv_param, filter_by_confidence

import torch
import torch.nn as nn

class Conv2DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build()

        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru
        self.dropout = config.dropout

        if config.pool_size: out_freq = config.num_bins // (2 ** config.num_layers)
        else: out_freq = config.num_bins
        self.gru = nn.GRU(input_size=self.params[-1]['output_channel']*out_freq, hidden_size=self.hidden_dim, num_layers=self.num_gru, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_classes)

    def build(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}', Conv2DBlock(param['input_channel'], param['output_channel'], 
                                                           kernel_size=self.kernel_size, padding='same', dilation=eval(self.dilation)))
            if self.config.cnn_dropout: enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size: enc.add_module(f'pool_{idx}', nn.MaxPool2d(eval(param['max_pool'])))
        return enc

    def forward(self, x): #  x.shape = b, 1, 80, 625
        if x.ndim == 3: x = x.unsqueeze(1)

        x = self.enc(x) # bcft
        b, _, _, t = x.shape # b,c,f,t (b, 256, f, 625)
        x = x.permute(0,3,1,2) # b,t,c,f
        x = x.reshape(b, t, -1) # b,t,cf

        x, _ = self.gru(x) # b,t,2h (b, 625, 512)
        x = self.fc(x) # b,t,num_classes (b, 625, 4)

        return x


class SegConv2DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = self.build()

        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru
        self.dropout = config.dropout

        if config.pool_size: out_freq = config.num_bins // (2 ** config.num_layers)
        else: out_freq = config.num_bins
        self.gru = nn.GRU(input_size=self.params[-1]['output_channel']*out_freq, hidden_size=self.hidden_dim, num_layers=self.num_gru, batch_first=True, dropout=self.dropout, bidirectional=True)

        if int(eval(config.pool_size)[1])==1:
            out_time = int(eval(config.num_frames))
        else:
            out_time = int(eval(config.num_frames)) // (2 ** config.num_layers)
        # self.fc = nn.Linear(self.hidden_dim*2*out_time, self.num_classes)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_classes)

    def build(self):
        enc = nn.Sequential()
        for idx, param in enumerate(self.params):
            enc.add_module(f'conv_{idx}', Conv2DBlock(param['input_channel'], param['output_channel'], 
                                                           kernel_size=self.kernel_size, padding='same', dilation=eval(self.dilation)))
            if self.config.cnn_dropout: enc.add_module(f'dropout_{idx}', nn.Dropout2d(self.config.cnn_dropout))
            if self.config.pool_size: enc.add_module(f'pool_{idx}', nn.MaxPool2d(eval(param['max_pool'])))
        return enc

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)
        x = self.enc(x) # bcft
        b, _, _, t = x.shape # b,c,f,t
        x = x.permute(0,3,1,2) # b,t,c,f
        x = x.reshape(b, t, -1) # b,t,cf
        x, _ = self.gru(x) # b,t,2h
        x = torch.max(x, dim=1)[0] # b, 2h*t
        # + Max pool, cnn gru n layer 수에 따라 실험
        x = self.fc(x) # b,num_classes
        return x