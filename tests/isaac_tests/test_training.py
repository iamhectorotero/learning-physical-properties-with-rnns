import unittest
import unittest.mock
import numpy as np
import torch
import io
import sys
from importlib import reload

from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy

# Disable TQDM for testing
from isaac import constants
constants.TQDM_DISABLE = True

from isaac import training
from isaac.models import initialise_model, MultiBranchModel


class TestTrainingLoop(unittest.TestCase):
    def setUp(self):
        sys.stderr = sys.__stderr__

    def tearDown(self):
        sys.stderr = sys.__stderr__

    @staticmethod
    def create_loader(n_features, n_classes, n_examples=10, seq_length=100, multibranch=False):

        X = np.random.rand(n_examples, seq_length, n_features).astype('f')

        if multibranch:
            Y = np.random.randint(low=0, high=n_classes, size=(n_examples, 2))
        else:
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
    def create_multibranch_model(n_features, n_classes):
        hidden_dim = 5
        network_params = (n_features, hidden_dim, n_classes)
        model, _, _ = initialise_model(network_params, arch=MultiBranchModel)
        return model

    @staticmethod
    def get_max_epochs_without_improvement(accuracies):
        max_epochs_without_improvement = 0
        epochs_without_improvement = 0
        best_accuracy = 0.

        for acc in accuracies:
            if acc > best_accuracy:
                best_accuracy = acc
                max_epochs_without_improvement = max(max_epochs_without_improvement,
                                                     epochs_without_improvement)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        max_epochs_without_improvement = max(max_epochs_without_improvement,
                                             epochs_without_improvement)

        return max_epochs_without_improvement

    def test_positive_patience(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = 2

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)
        max_epochs_without_improvement = self.get_max_epochs_without_improvement(accuracies[1])
        # The training stops when the number of epochs without improvement surpasses the patience
        self.assertGreaterEqual(patience + 1, max_epochs_without_improvement)

    def test_zero_patience(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = 0

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)
        max_epochs_without_improvement = self.get_max_epochs_without_improvement(accuracies[1])
        # The first time the accuracy doesn't improve the training stops
        self.assertGreaterEqual(1, max_epochs_without_improvement)

    def test_negative_patience(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = -2

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)
        max_epochs_without_improvement = self.get_max_epochs_without_improvement(accuracies[1])
        # Negative patience behaves equal to zero patience
        self.assertGreaterEqual(1, max_epochs_without_improvement)

    def test_positive_num_epochs(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        self.assertEqual(len(losses), num_epochs)
        self.assertEqual(len(accuracies[0]), num_epochs)
        self.assertEqual(len(accuracies[1]), num_epochs)

    def test_zero_num_epochs(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 0
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        self.assertEqual(len(losses), num_epochs)
        self.assertEqual(len(accuracies[0]), num_epochs)
        self.assertEqual(len(accuracies[1]), num_epochs)

    def test_negative_num_epochs(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = -1
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        # Negative epochs behaves like 0 epochs
        self.assertEqual(len(losses), 0)
        self.assertEqual(len(accuracies[0]), 0)
        self.assertEqual(len(accuracies[1]), 0)

    def test_model_is_fitted(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        initial_model = deepcopy(model)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        # Assumes the accuracy of the model will improve
        for p1, p2 in zip(initial_model.parameters(), best_model.parameters()):
            self.assertFalse(torch.eq(p1.data, p2.data).any())

    def test_error_improves_with_only_one_class_in_the_training_set(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)

        # Setting a small learning rate to ensure that improvement happens over the span
        # of multiple epochs
        optimizer = Adam(model.parameters(), lr=1e-8)
        error = CrossEntropyLoss()

        # Setting the number of training classes to 1, overfitting is easy and error should
        # necessarily improve
        train_loader = self.create_loader(n_features, 1)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        self.assertNotEqual(losses[0], min(losses))

    def test_error_improves_with_more_than_one_class_in_the_training_set(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        # Even with classes randomly assigned, the model should be able to overfit to the training
        # set even if this doesn't imply improvement in the validation
        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        self.assertNotEqual(losses[0], min(losses))
        # Accuracy in the training set should be better than random
        self.assertGreaterEqual(max(accuracies[0]), 1./n_classes)

    def test_model_with_best_val_accuracy_is_returned(self):
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        initial_model = deepcopy(model)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        best_model_accuracy = training.evaluate(best_model, val_loader)
        self.assertEqual(max(accuracies[1]), best_model_accuracy)

    def test_model_with_best_val_accuracy_is_returned_for_multibranch_model(self):
        n_features = 4
        n_classes = 3
        model = self.create_multibranch_model(n_features, n_classes)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes, multibranch=True)
        val_loader = self.create_loader(n_features, n_classes, multibranch=True)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience, multibranch=True)

        best_first_accuracy = training.evaluate(best_model[0], val_loader)
        self.assertEqual(max(np.array(accuracies[1])[:, 0]), best_first_accuracy[0])
        best_second_accuracy = training.evaluate(best_model[1], val_loader)
        self.assertEqual(max(np.array(accuracies[1])[:, 1]), best_second_accuracy[1])

    def test_returned_statistics_shape_is_correct_for_multibranch_model(self):
        n_features = 4
        n_classes = 3
        model = self.create_multibranch_model(n_features, n_classes)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes, multibranch=True)
        val_loader = self.create_loader(n_features, n_classes, multibranch=True)
        num_epochs = 10
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience, multibranch=True)

        self.assertEqual(losses.shape[1], 2)
        self.assertEqual(len(losses[:, 0]), num_epochs)
        self.assertEqual(len(losses[:, 1]), num_epochs)

        self.assertEqual(len(accuracies[0][:, 0]), num_epochs)
        self.assertEqual(len(accuracies[0][:, 1]), num_epochs)
        self.assertEqual(len(accuracies[1][:, 0]), num_epochs)
        self.assertEqual(len(accuracies[1][:, 1]), num_epochs)

    @unittest.mock.patch('sys.stderr', new_callable=io.StringIO)
    def test_print_stats_per_epoch_is_true(self, mock_stderr):
        # To test if a method prints stats in TQDM re-enable TQDM printing
        print(mock_stderr.getvalue().count("Train_loss"))

        constants.TQDM_DISABLE = False
        reload(training)

        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        initial_model = deepcopy(model)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 1
        print_stats_per_epoch = True

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)
        print(mock_stderr.getvalue().count("Train_loss"))

        # Stats are printed once more at the end of the loop
        self.assertEqual(mock_stderr.getvalue().count("Train_loss"), num_epochs + 1)
        self.assertEqual(mock_stderr.getvalue().count("Train_acc"), num_epochs + 1)
        self.assertEqual(mock_stderr.getvalue().count("Val_acc"), num_epochs + 1)

        # At the end of the method testing, disable TQDM printing again
        constants.TQDM_DISABLE = True
        reload(training)

    @unittest.mock.patch('sys.stderr', new_callable=io.StringIO)
    def test_print_stats_per_epoch_is_false(self, mock_stderr):
        # To test if a method prints stats in TQDM re-enable TQDM printing
        constants.TQDM_DISABLE = False
        reload(training)

        sys.stderr = sys.__stderr__
        n_features = 4
        n_classes = 3
        model = self.create_model(n_features, n_classes)
        initial_model = deepcopy(model)

        optimizer = Adam(model.parameters())
        error = CrossEntropyLoss()

        train_loader = self.create_loader(n_features, n_classes)
        val_loader = self.create_loader(n_features, n_classes)
        num_epochs = 1
        print_stats_per_epoch = False

        seq_start, seq_end, seq_step = None, None, None
        patience = np.inf

        losses, accuracies, best_model = training.training_loop(model, optimizer, error,
                                                                train_loader, val_loader,
                                                                num_epochs, print_stats_per_epoch,
                                                                seq_start, seq_end, seq_step,
                                                                patience)

        self.assertEqual(mock_stderr.getvalue().count("Train_loss"), 0)
        self.assertEqual(mock_stderr.getvalue().count("Train_acc"), 0)
        self.assertEqual(mock_stderr.getvalue().count("Val_acc"), 0)

        # At the end of the method testing, disable TQDM printing again
        constants.TQDM_DISABLE = True
        reload(training)


if __name__ == "__main__":
    unittest.main()
