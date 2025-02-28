import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes, num_heads, num_layers, hidden_dim ):
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

class CNNModel(nn.Module):
    def __init__(self, input_channel, num_classes, kernel_size, num_filters):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv1d(input_channel, num_filters, kernel_size = kernel_size, padding = 1)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size = kernel_size, padding = 1)
        self.conv3 = nn.Conv1d(num_filters, num_filters * 4, kernel_size = kernel_size, padding  =  1)
        self.pool = nn.MaxPool1d(kernel_size = 2, stride = 2)

        self.fc1 =nn.Linear(num_filters *4, 128)
        self.fc_mode = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.mean(dim = -1)
        x = F.relu(self.fc(x))
        x_mode = self.fc_mode(x)

        return x_mode 

def detect_mode_change(ujo_prob, gyemyeonjo_prob, threshold=0.05):
    mode_change = (np.abs(ujo_prob - gyemyeonjo_prob) < threshold).astype(int)
    return mode_change

    