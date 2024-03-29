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

    def predict_seq2seq_in_rolling_windows(self, x, seconds_per_window):

        output_seq = None

        # frames_per_interval = INTERVAL_SIZE * FPS // STEP_SIZE
        frames_per_second = 1 * 60 // 3
        window_size = frames_per_second*seconds_per_window

        for i in range(0, x.shape[1]-window_size, frames_per_second):

            out, _ = self.rec_layer(x[:, i:i+window_size, :])
            out = self.fc(out)[:, -1:, :]

            if output_seq is None:
                output_seq = out
            else:
                output_seq = torch.cat([output_seq, out], dim=1)
        return output_seq


class MultiBranchModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5, cell_type=nn.GRU):
        super(MultiBranchModel, self).__init__()

        self.layers_per_section = 2

        self.base_gru = cell_type(input_dim, hidden_dim, self.layers_per_section, batch_first=True, dropout=dropout)

        self.gru1 = cell_type(hidden_dim, hidden_dim, self.layers_per_section, batch_first=True, dropout=dropout)
        self.gru2 = cell_type(hidden_dim, hidden_dim, self.layers_per_section, batch_first=True, dropout=dropout)

        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.base_gru(x)

        out_1, _ = self.gru1(out)
        out_1 = self.fc1(out_1[:, -1, :])

        out_2, _ = self.gru2(out)
        out_2 = self.fc2(out_2[:, -1, :])

        return out_1, out_2

    def predict_seq2seq_in_intervals(self, x):

        output_seq1 = None
        output_seq2 = None

        # frames_per_interval = INTERVAL_SIZE * FPS // STEP_SIZE
        frames_per_interval = 1 * 60 // 3

        for i in range(0, x.shape[1], frames_per_interval):

            out, _ = self.base_gru(x[:, i:i+frames_per_interval, :])

            out_1, _ = self.gru1(out)
            out_1 = self.fc1(out_1)[:, -1:, :]

            out_2, _ = self.gru2(out)
            out_2 = self.fc2(out_2)[:, -1:, :]

            if output_seq1 is None:
                output_seq1 = out_1
                output_seq2 = out_2
            else:
                output_seq1 = torch.cat([output_seq1, out_1], dim=1)
                output_seq2 = torch.cat([output_seq2, out_2], dim=1)

        return output_seq1, output_seq2

    def predict_seq2seq_in_rolling_windows(self, x, seconds_per_window):

        output_seq1 = None
        output_seq2 = None

        # frames_per_interval = INTERVAL_SIZE * FPS // STEP_SIZE
        frames_per_second = 1 * 60 // 3
        window_size = frames_per_second*seconds_per_window

        for i in range(0, x.shape[1]-window_size, frames_per_second):

            out, _ = self.base_gru(x[:, i:i+window_size, :])

            out_1, _ = self.gru1(out)
            out_1 = self.fc1(out_1)[:, -1:, :]

            out_2, _ = self.gru2(out)
            out_2 = self.fc2(out_2)[:, -1:, :]

            if output_seq1 is None:
                output_seq1 = out_1
                output_seq2 = out_2
            else:
                output_seq1 = torch.cat([output_seq1, out_1], dim=1)
                output_seq2 = torch.cat([output_seq2, out_2], dim=1)

        return output_seq1, output_seq2


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
