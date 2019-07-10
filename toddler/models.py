import torch.nn as nn
import numpy as np
import torch


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.):
        super(ValueNetwork, self).__init__()
        # RNN
        # self.project = nn.Linear(input_dim, hidden_dim)
        # self.relu = nn.ReLU()
        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        # Readout layer
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.hh = None

    def forward(self, x, online=False):
        # x = self.project(x)
        # x = self.relu(x)
        if online:
            out, self.hh = self.rec_layer(x, self.hh)
        else:
            out, _ = self.rec_layer(x)
        out = self.readout(out)
        return out

    def reset_hidden_states(self):
        self.hh = None
