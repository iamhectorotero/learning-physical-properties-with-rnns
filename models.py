import torch.nn as nn

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
    def __init__(self, input_dim, first_hidden_dim, second_hidden_dim, output_dim):
        super(ComplexRNNModel, self).__init__()
        # RNN
        self.lstm = nn.LSTM(input_dim, first_hidden_dim, 1, batch_first=True, )
        self.lstm_2 = nn.LSTM(first_hidden_dim, second_hidden_dim, 1, batch_first=True)
        
        # Readout layer
        self.fc = nn.Linear(second_hidden_dim, output_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out, _ = self.lstm_2(out)
        out = self.fc(out[:, -1, :]) 
        return out