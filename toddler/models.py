import torch.nn as nn
import numpy as np
import torch


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.0):
        super(ValueNetwork, self).__init__()
        # self.project = nn.Linear(input_dim, input_dim)
        # self.relu = nn.ReLU()
        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.hh = None

    def forward(self, x):
        # x = self.relu(self.project(x))
        out, self.hh = self.rec_layer(x, self.hh)
        out = self.readout(out)
        return out

    def reset_hidden_states(self, hh=None):
        self.hh = hh


class DistributedValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.0, n_processes=10):
        super(DistributedValueNetwork, self).__init__()
        self.project = nn.Linear(input_dim, input_dim)
        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.hh = [None for _ in range(n_processes)]

    def forward(self, x, process_i):
        x = self.project(x)
        out, self.hh[process_i] = self.rec_layer(x, self.hh)
        out = self.readout(out)
        return out

    def reset_hidden_states(self, hh=None):
        self.hh[process_i] = hh[process_i]


class Policy(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0., n_processes=1):
        super(Policy, self).__init__()
        self.rec_layer = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.hh = [None for _ in range(n_processes)]
        self.policy_log_probs = []

    def forward(self, x, idx):
        out, self.hh[idx] = self.rec_layer(x, self.hh[idx])
        out = self.readout(out)
        out = self.softmax(out)
        return out

    def reset_hidden_states(self, idx):
        self.hh[idx] = None

class NonRecurrentValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, dropout=0.):
        super(NonRecurrentValueNetwork, self).__init__()

        self.n_layers = n_layers
        self.layers = nn.ModuleDict()
        for i in range(n_layers):
            self.layers["fc_"+str(i)] = nn.Linear(input_dim, hidden_dim)
            self.layers["act_"+str(i)] = nn.ReLU()
        self.layers["fc_"+str(n_layers)] = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = x
        for i in range(self.n_layers):
            out = self.layers["fc_"+str(i)](out)
            out = self.layers["act_"+str(i)](out)
        out = self.layers["fc_"+str(self.n_layers)](out)
        return out

