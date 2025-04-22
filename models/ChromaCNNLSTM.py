from .modules import ConvBlock

import torch.nn as nn

class ChromaCNNLSTM(nn.Module):
    def __init__(self, conv_layers, conv_in_channels, pool, fc_layers, fc_in_channels, dropout, num_classes, num_bins=25):
        super().__init__()
        self.conv_layers = self.get_conv_layers(conv_layers, conv_in_channels, pool)
        self.lstm = nn.LSTM(input_size=64*(int(num_bins)//4), hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_layers = self.get_fc_layers(fc_layers, fc_in_channels, dropout, num_classes)

    def get_conv_layers(self, conv_layers, in_channels, pool):
        layers = []
        for val in conv_layers:
            if val=="M": 
                layers.append(nn.MaxPool2d(kernel_size=eval(pool)))
            else: 
                layers.append(ConvBlock(in_channels=in_channels, out_channels=val))
                in_channels = val
        return nn.ModuleList(layers)

    def get_fc_layers(self, fc_layers, in_channels, dropout, num_classes):
        layers = []
        in_channels  = eval(in_channels)
        for val in fc_layers:
            if val=="D": layers.append(nn.Dropout(dropout))
            elif val=="relu": layers.append(nn.ReLU())
            elif val=="gelu": layers.append(nn.GELU())
            else:
                layers.append(nn.Linear(in_features=in_channels, out_features=val))
                in_channels = val
        layers.append(nn.Linear(in_features=in_channels, out_features=num_classes))
        return nn.ModuleList(layers)
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        for layer in self.conv_layers: x = layer(x)
        b, c, f, t = x.shape
        x = x.permute(0,3,1,2)
        x = x.reshape(b, t, -1)
        x, _ = self.lstm(x)
        for layer in self.fc_layers: x = layer(x)
        return x