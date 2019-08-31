import torch.nn as nn
import numpy as np
import torch

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.0):
        super(ValueNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.hh = None

    def forward(self, x):
        out, hiddens = self.rec_layer(x, self.hh)
        self.hh = repackage_hidden(hiddens)
        out = self.readout(out)
        return out

    def predict(self, x, hiddens):
        out, _ = self.rec_layer(x, hiddens)
        out = self.readout(out)
        return out

    def reset_hidden_states(self, hh=None):
        self.hh = hh

