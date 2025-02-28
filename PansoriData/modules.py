import numpy as np
import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_length, num_classes, device):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) 
        self.fc = nn.Linear(seq_length*hidden_size * 2, num_classes)

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out.reshape(-1,self.seq_length*self.hidden_size * 2) 
        out = self.fc(out)
        return out
    
