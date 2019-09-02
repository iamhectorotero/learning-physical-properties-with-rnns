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


def initialise_model(network_params, lr=0.01, seed=0, arch=ComplexRNNModel, cell_type=nn.GRU,
                     device=torch.device("cpu")):
    """Initialises a model, error and optimizer with the specified parameters.
    Args:
        network_params: an iterable containing the network parameters that will be unpacked and
        passed to architecture class.

        lr: Adam optimizer's learning rate.

        seed: numpy's and torch random seed will be set to this value.

        arch: a class extending torch.nn.Module implementing the neural network architecture. Must
        accept the network_params and an extra cell_type. Defaults to ComplexRNNModel.

        cell_type: RNN cell type (eg. nn.GRU, nn.LSTM) that will be passed to the architecture.
        Defaults to GRU.

        device: a torch.device to choose where the model will be executed. Defaults to CPU.
    Returns:
        model: initialised instance of the neural network.

        error: nn.CrossEntropyLoss (sent to the specified device)

        optimizer: torch.optim.Adam with the model parameters and learning rate specified.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = arch(*network_params, cell_type=cell_type)
    model = model.to(device=device)

    error = nn.CrossEntropyLoss().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model, error, optimizer
