import torch.nn as nn
import numpy as np
import torch


class ComplexRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0., cell_type=nn.GRU):
        super(ComplexRNNModel, self).__init__()
        self.rec_layer = cell_type(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rec_layer(x)
        out = self.fc(out[:, -1, :]) 
        return out


def initialise_model(network_params, lr=0.01, seed=0, arch=ComplexRNNModel, cell_type=nn.GRU):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = arch(*network_params, cell_type=cell_type)
    model = model.cuda()

    error = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    return model, error, optimizer
