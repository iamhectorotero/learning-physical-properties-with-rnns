import unittest
from isaac import models 
import torch
import os
import warnings

class TestInitialiseModel(unittest.TestCase):
    def test_for_model_params(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1

        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)
        model, _, _ = models.initialise_model(network_params)

        self.assertEqual(model.rec_layer.input_size, input_size)
        self.assertEqual(model.rec_layer.hidden_size, hidden_dim)
        self.assertEqual(model.rec_layer.num_layers, n_layers)
        self.assertEqual(model.fc.out_features, output_dim)
        self.assertEqual(model.rec_layer.dropout, dropout)

    def test_for_lr(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)
        lr = 0.0001
        _, _, optimizer= models.initialise_model(network_params, lr=lr)
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], lr)

    def test_for_reproducibility(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

        first_model, _, _ = models.initialise_model(network_params, seed=0)
        second_model, _, _ = models.initialise_model(network_params, seed=0)

        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            self.assertTrue(torch.allclose(first_param, second_param))

    def test_for_different_initialisations(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

        first_model, _, _ = models.initialise_model(network_params, seed=0)
        second_model, _, _ = models.initialise_model(network_params, seed=72)

        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            self.assertFalse(torch.allclose(first_param, second_param))

    def test_default_architecture(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)
        model, _, _ = models.initialise_model(network_params)
        self.assertTrue(isinstance(model, models.ComplexRNNModel))

    def test_alternative_architecture(self):
        class AlternativeArch(torch.nn.Module):
            def __init__(self, cell_type):
                super(AlternativeArch, self).__init__()
                self.rec_layer = cell_type(2, 2, 2, False)
            def forward(self, x):
                return self.rec_layer(x)[0]

        network_params = []
        model, _, _ = models.initialise_model(network_params, arch=AlternativeArch)
        self.assertTrue(isinstance(model, AlternativeArch))

    def test_for_cell_type(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)
        model, _, _ = models.initialise_model(network_params)
        self.assertTrue(isinstance(model.rec_layer, torch.nn.GRU))

        model, _, _ = models.initialise_model(network_params, cell_type=torch.nn.RNN)
        self.assertTrue(isinstance(model.rec_layer, torch.nn.RNN))

    def test_for_GPU_device(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

        if os.environ.get("CUDA_DEVICES_AVAILABLE", default="") != "":
            gpu_device = torch.device("cuda:0")
            model, _, _ = models.initialise_model(network_params, device=gpu_device)
            for param in model.parameters():
                self.assertTrue(param.is_cuda)
        else:
            warnings.warn("No Cuda device available to run this test.")

    def test_for_CPU_device(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

        cpu_device = torch.device("cpu")
        model, _, _ = models.initialise_model(network_params, device=cpu_device)
        for param in model.parameters():
            self.assertFalse(param.is_cuda)

    def test_model_forward_output(self):
        input_size = 2
        hidden_dim = 3
        n_layers = 4
        output_dim = 5
        dropout = 0.1
        network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

        model, _, _ = models.initialise_model(network_params)
        input_tensor = torch.arange(10 * 8 * 2, dtype=torch.float).view(10, 8, 2)
        expected_output_shape = [10, 5]
        output = model(input_tensor)

        self.assertListEqual(list(output.shape), expected_output_shape)

if __name__ == "__main__":
    unittest.main()