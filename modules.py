import numpy as np
import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads = 4, num_layers = 2, hidden_dim = 128):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model = input_size, nhead = num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
        self.fc_mode = nn.Linear(input_size, num_classes)
        self.fc_modulation = nn.Linear(input_size, 1)
    
    def forward(self, x):
        x = self.transformer_encoder(x)
        x_mode = self.fc_mode(x.mean(dim = 1))
        x_modulation = torch.sigmoid(self.fc.modulation(x))
        return x_mode, x_modulation.squeeze(-1)
    

    