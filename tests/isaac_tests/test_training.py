import unittest
import numpy as np
import torch

from isaac import training
from isaac.models import initialise_model

class TestEvaluate(unittest.TestCase):
    @staticmethod
    def create_loader(n_features, n_classes, n_examples=10, seq_length=100):

        X = np.random.rand(n_examples, seq_length, n_features).astype('f')
        Y = np.random.randint(low=0, high=n_classes, size=n_examples)

        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        tensor_dataset = torch.utils.data.TensorDataset(X, Y)
        val_loader = torch.utils.data.DataLoader(tensor_dataset, shuffle=False)

        return val_loader

    @staticmethod
    def create_model(n_features, n_classes):
        hidden_dim = 5
        n_layers = 1
        network_params = (n_features, hidden_dim, n_layers, n_classes)
        model, _, _ = initialise_model(network_params)
        return model

    @staticmethod
    def is_accuracy_well_calculated(accuracy, val_loader, predicted):
        classes = np.concatenate([y.numpy() for _, y in val_loader])

        correct = 0
        for y_pred, y_true in zip(predicted, classes):
            correct += int(y_pred == y_true)
        calculated_accuracy = 100 * correct / float(len(classes))

        return np.isclose(accuracy, calculated_accuracy)

    def test_accuracy_is_correct(self):
        n_features = 10
        n_classes = 3
        n_examples = 10
        seq_start, seq_end, step_size = None, None, None

        val_loader = self.create_loader(n_features, n_classes, n_examples)
        model = self.create_model(n_features, n_classes)

        return_predicted = True
        accuracy, predicted = training.evaluate(model, val_loader, return_predicted,
                                                seq_start, seq_end, step_size)

        self.assertEqual(len(predicted), n_examples)
        self.assertTrue(self.is_accuracy_well_calculated(accuracy, val_loader, predicted))

    def test_perfect_accuracy(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_start, seq_end, step_size = None, None, None

        val_loader = self.create_loader(n_features, n_classes, n_examples)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        accuracy = training.evaluate(model, val_loader, return_predicted,
                                                seq_start, seq_end, step_size)

        self.assertEqual(accuracy, 100.0)

    def test_step_size_different_from_none(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        seq_start, seq_end, step_size = None, None, 2

        training.evaluate(model, val_loader, return_predicted,
                          seq_start, seq_end, step_size)

    def test_step_size_larger_than_seq_length(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False

        # In these cases only the first step of every example will be taken
        seq_start, seq_end, step_size = None, None, seq_length + 2
        training.evaluate(model, val_loader, return_predicted,
                          seq_start, seq_end, step_size)


    def test_passing_seq_start_negative(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False

        # If seq_start is negative, the indexing will start from the end
        seq_start, seq_end, step_size = -1, None, None
        training.evaluate(model, val_loader, return_predicted,
                          seq_start, seq_end, step_size)

    def test_passing_seq_start_larger_than_seq_length(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        seq_start, seq_end, step_size = seq_length + 25, None, None

        # Passing an empty sequence should generate an error
        with self.assertRaises(RuntimeError):
            training.evaluate(model, val_loader, return_predicted,
                              seq_start, seq_end, step_size)

    def test_passing_seq_end_larger_than_seq_length(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        seq_start, seq_end, step_size = None, seq_length + 10, None

        # Sequences will be loaded fully if the seq_end is larger then seq_length
        training.evaluate(model, val_loader, return_predicted,
                          seq_start, seq_end, step_size)

    def test_passing_seq_start_larger_than_seq_end(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        seq_start = 10
        seq_end = seq_start - 2
        step_size = None

        # Passing an empty sequence should generate an error
        with self.assertRaises(RuntimeError):
            training.evaluate(model, val_loader, return_predicted,
                              seq_start, seq_end, step_size)

    def test_passing_seq_end_negative(self):
        n_features = 10
        n_classes = 1
        n_examples = 10
        seq_length = 100

        val_loader = self.create_loader(n_features, n_classes, n_examples, seq_length)
        model = self.create_model(n_features, n_classes)

        return_predicted = False
        seq_start, seq_end, step_size = None, -10, None

        # If seq_end is negative, the indexing will happen from the back
        training.evaluate(model, val_loader, return_predicted,
                          seq_start, seq_end, step_size)


if __name__ == "__main__":
    unittest.main()
