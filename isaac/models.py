import torch.nn as nn
import numpy as np
import torch

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        # Number of hidden dimensions
        self.hidden_dim = hidden_dim
        
        # Number of hidden layers
        self.layer_dim = layer_dim
        
        # RNN
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :]) 
        return out
    
    
class ComplexRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.):
        super(ComplexRNNModel, self).__init__()
        # RNN
        self.rec_one = nn.GRU(input_dim, hidden_dim, n_layers // 2, batch_first=True, dropout=dropout)
        self.rec_two = nn.GRU(hidden_dim, hidden_dim, n_layers // 2, batch_first=True, dropout=dropout)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.rec_one(x)
        out, _ = self.rec_two(out)
        out = self.fc(out[:, -1, :]) 
        return out
    
    
def initialise_model(network_dims, dropout=None, lr=0.01, seed=0, arch=ComplexRNNModel):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = arch(*network_dims, dropout=dropout)
    model = model.cuda()

    error = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, error, optimizer