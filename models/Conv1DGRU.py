from models.modules import Conv1DBlock
from models.model_utils import calc_conv_param, filter_by_confidence

import torch.nn as nn

class Conv1DGRU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.params = calc_conv_param(config)

        self.kernel_size = config.kernel_size
        self.dilation = config.dilation
        self.num_classes = config.num_classes
        self.enc = nn.Sequential()
        self.build()

        self.hidden_dim = config.hidden_dim
        self.num_gru = config.num_gru
        self.dropout = config.dropout
        self.gru = nn.GRU(input_size=self.params[-1]['output_channel'], hidden_size=self.hidden_dim, num_layers=self.num_gru, batch_first=True, dropout=self.dropout, bidirectional=True)
        self.fc = nn.Linear(self.hidden_dim*2, self.num_classes)

    def build(self):
        for idx, param in enumerate(self.params):
            self.enc.add_module(f'conv_{idx}', Conv1DBlock(param['input_channel'], param['output_channel'], 
                                                           kernel_size=self.kernel_size, padding='same', dilation=self.dilation))
            if self.config.pool_size: self.enc.add_module(f'pool_{idx}', nn.MaxPool1d(param['max_pool']))
    
    def forward(self, x):
        x = filter_by_confidence(x) # b,c,t
        x = self.enc(x) # b,c,t
        x = x.permute(0, 2, 1) # b,t,c
        x, _ = self.gru(x) # b,t,2h
        x = self.fc(x) # b,t,num_classes
        return x