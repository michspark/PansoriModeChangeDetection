import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, num_layers, hidden_dim, input_size, num_classes, dropout):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        x, _ = self.gru(x.permute(0,2,1))
        x = self.fc(x)
        return x