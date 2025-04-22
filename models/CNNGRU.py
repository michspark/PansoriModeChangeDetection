from models.modules import ConvBlock

import torch.nn as nn

class CNNGRU(nn.Module):
    def __init__(self, first_conv, num_conv, max_pool, num_bins, hidden_dim, dropout, num_gru, num_classes):
        super().__init__()
        self.conv_layers = self.get_conv_layers(first_conv, num_conv, max_pool)
        conv_out = first_conv * (2 ** (num_conv - 1))
        if max_pool: conv_freq = num_bins // (2 ** num_conv)
        else: conv_freq = num_bins    

        gru_in = conv_out * conv_freq
        self.gru = nn.GRU(input_size=gru_in, hidden_size=hidden_dim, num_layers=num_gru, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def get_conv_layers(self, first_conv, num_conv, max_pool):
        conv_layers = []
        conv_in = 1
        for val in range(num_conv):
            conv_out = first_conv*(2**val)
            conv_layers.append(ConvBlock(in_channels=conv_in, out_channels=conv_out))
            if max_pool: conv_layers.append(nn.MaxPool2d(kernel_size=(2,1)))
            conv_in = conv_out
        return  nn.ModuleList(conv_layers)

    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)

        for layer in self.conv_layers:
            x = layer(x)

        b, _, _, t = x.shape # b,c,f,t
        x = x.permute(0,3,1,2) # b,t,c,f
        x = x.reshape(b, t, -1) # b,t,cf

        x, _ = self.gru(x) # b,t,2h

        x = self.fc(x) # b,t,num_classes
        return x