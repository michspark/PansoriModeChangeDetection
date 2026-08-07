import torch 
import torch.nn as nn
from torch.nn.utils import weight_norm 

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv_norm = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv_norm(x)

