import torch.nn as nn
import numpy as np
import torch


class ComplexRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.):
        super(ComplexRNNModel, self).__init__()
        # RNN
        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # Readout layer
        self.dropout = nn.Dropout(p=dropout/2)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.rec_layer(x)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out) 
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