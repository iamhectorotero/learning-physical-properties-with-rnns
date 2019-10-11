import unittest
from isaac import models 
import torch
import os
import warnings

class TestInitialiseModel(unittest.TestCase):
    input_size = 2
    hidden_dim = 3
    n_layers = 4
    output_dim = 5
    dropout = 0.1

    network_params = (input_size, hidden_dim, n_layers, output_dim, dropout)

    def test_for_model_params_for_default_architecture(self):
        model, _, _ = models.initialise_model(self.network_params)

        self.assertEqual(model.rec_layer.input_size, self.input_size)
        self.assertEqual(model.rec_layer.hidden_size, self.hidden_dim)
        self.assertEqual(model.rec_layer.num_layers, self.n_layers)
        self.assertEqual(model.fc.in_features, self.hidden_dim)
        self.assertEqual(model.fc.out_features, self.output_dim)
        self.assertEqual(model.rec_layer.dropout, self.dropout)

    def test_for_model_params_for_multibranch_architecture(self):
        network_params = (self.input_size, self.hidden_dim, self.output_dim, self.dropout)
        model, _, _ = models.initialise_model(network_params, arch=models.MultiBranchModel)

        self.assertEqual(model.base_gru.input_size, self.input_size)
        self.assertEqual(model.base_gru.hidden_size, self.hidden_dim)

        self.assertEqual(model.gru1.input_size, self.hidden_dim)
        self.assertEqual(model.gru1.hidden_size, self.hidden_dim)
        self.assertEqual(model.gru2.input_size, self.hidden_dim)
        self.assertEqual(model.gru2.hidden_size, self.hidden_dim)

        self.assertEqual(model.fc1.in_features, self.hidden_dim)
        self.assertEqual(model.fc1.out_features, self.output_dim)
        self.assertEqual(model.fc2.in_features, self.hidden_dim)
        self.assertEqual(model.fc2.out_features, self.output_dim)

        self.assertEqual(model.base_gru.dropout, self.dropout)
        self.assertEqual(model.gru1.dropout, self.dropout)
        self.assertEqual(model.gru2.dropout, self.dropout)

    def test_for_lr(self):
        lr = 0.0001
        _, _, optimizer= models.initialise_model(self.network_params, lr=lr)
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], lr)

    def test_for_reproducibility(self):
        first_model, _, _ = models.initialise_model(self.network_params, seed=0)
        second_model, _, _ = models.initialise_model(self.network_params, seed=0)

        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            self.assertTrue(torch.allclose(first_param, second_param))

    def test_for_different_initialisations(self):
        first_model, _, _ = models.initialise_model(self.network_params, seed=0)
        second_model, _, _ = models.initialise_model(self.network_params, seed=72)

        for first_param, second_param in zip(first_model.parameters(), second_model.parameters()):
            self.assertFalse(torch.allclose(first_param, second_param))

    def test_default_architecture(self):
        model, _, _ = models.initialise_model(self.network_params)
        self.assertTrue(isinstance(model, models.ComplexRNNModel))

    def test_alternative_architecture(self):
        network_params = (self.input_size, self.hidden_dim, self.output_dim, self.dropout)

        model, _, _ = models.initialise_model(network_params, arch=models.MultiBranchModel)
        self.assertTrue(isinstance(model, models.MultiBranchModel))

    def test_for_cell_type(self):
        model, _, _ = models.initialise_model(self.network_params)
        self.assertTrue(isinstance(model.rec_layer, torch.nn.GRU))

        model, _, _ = models.initialise_model(self.network_params, cell_type=torch.nn.RNN)
        self.assertTrue(isinstance(model.rec_layer, torch.nn.RNN))

    def test_for_GPU_device(self):
        if os.environ.get("CUDA_DEVICES_AVAILABLE", default="") != "":
            gpu_device = torch.device("cuda:0")
            model, _, _ = models.initialise_model(self.network_params, device=gpu_device)
            for param in model.parameters():
                self.assertTrue(param.is_cuda)
        else:
            warnings.warn("No Cuda device available to run this test.")

    def test_for_CPU_device(self):
        cpu_device = torch.device("cpu")
        model, _, _ = models.initialise_model(self.network_params, device=cpu_device)
        for param in model.parameters():
            self.assertFalse(param.is_cuda)

    def test_model_forward_output(self):
        model, _, _ = models.initialise_model(self.network_params)
        input_tensor = torch.arange(10 * 8 * 2, dtype=torch.float).view(10, 8, 2)
        expected_output_shape = [10, 5]
        output = model(input_tensor)

        self.assertListEqual(list(output.shape), expected_output_shape)

    def test_model_forward_output_for_multibranch_arch(self):
        network_params = (self.input_size, self.hidden_dim, self.output_dim, self.dropout)
        model, _, _ = models.initialise_model(network_params, arch=models.MultiBranchModel)

        input_tensor = torch.arange(10 * 8 * 2, dtype=torch.float).view(10, 8, 2)
        expected_output_shape = [10, 5]
        first_output, second_output = model(input_tensor)

        self.assertListEqual(list(first_output.shape), expected_output_shape)
        self.assertListEqual(list(second_output.shape), expected_output_shape)


if __name__ == "__main__":
    unittest.main()
