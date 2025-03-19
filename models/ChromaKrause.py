from .modules import KrauseConvBlock2D, KrauseConvBlock1D
import torch.nn as nn

class ChromaKrause(nn.Module):
    def __init__(self,  conv_layers_1, conv_layers_2, conv_layers_3, conv_layers_4, conv_layers_5, conv_in_channels, pool, dilation, fc_layers, fc_in_channels, dropout, num_classes):
        super().__init__()

        self.dilation_idx = 0

        self.conv_layers_1 = self.get_conv_layers(conv_layers_1, conv_in_channels, pool, dilation, conv_type="2D")
        self.conv_layers_2 = self.get_conv_layers(conv_layers_2, conv_layers_1[-2], pool, dilation, conv_type="2D")
        self.conv_layers_3 = self.get_conv_layers(conv_layers_3, conv_layers_2[-2], pool, dilation, conv_type="2D")
        self.conv_layers_4 = self.get_conv_layers(conv_layers_4, conv_layers_3[-2], pool, dilation, conv_type="2D")
        self.conv_layers_5 = self.get_conv_layers(conv_layers_5, conv_layers_4[-2], pool, dilation, conv_type="1D")

        #self.lstm = nn.LSTM(input_size = 256*(25//8), hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        self.fc_layers = self.get_fc_layers(fc_layers, fc_in_channels, dropout, num_classes)

    def get_conv_layers(self, conv_layers, in_channels, pool, dilation, conv_type = '2D'):
        layers = []
        for val in conv_layers:

            if val=="M":
                layers.append(nn.MaxPool2d(kernel_size=eval(pool)) if conv_type == "2D" else nn.MaxPool1d(kernel_size=3, stride=1, padding = 1))
                continue

            elif val == "S":
                layers.append(nn.AdaptiveAvgPool2d((1, None)))
                continue

            else:
                dilation_val = dilation[self.dilation_idx] if isinstance(dilation, list) else dilation
                if conv_type == '2D':
                    layers.append(KrauseConvBlock2D(in_channels=in_channels, out_channels=val, kernel_size=3, stride=1, dilation=int(dilation_val)))
                else:
                    layers.append(KrauseConvBlock1D(in_channels=in_channels, out_channels=val, kernel_size=3, stride=1, dilation=int(dilation_val)))

                in_channels = val

                self.dilation_idx += 1

        return nn.ModuleList(layers)

    def get_fc_layers(self, fc_layers, in_channels, dropout, num_classes):
        layers = []
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

        for layer in self.conv_layers_1: x = layer(x)
        for layer in self.conv_layers_2: x = layer(x)
        for layer in self.conv_layers_3: x = layer(x)
        for layer in self.conv_layers_4: x = layer(x)
        x = x.squeeze(2)
        for layer in self.conv_layers_5: x = layer(x)

        b, c, t = x.shape
        x = x.permute(0,2,1)
        x = x.reshape(b, t, -1)

        #x, _ = self.lstm(x)

        for layer in self.fc_layers: x = layer(x)

        return x