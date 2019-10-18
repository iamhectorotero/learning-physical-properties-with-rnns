import unittest
import unittest.mock
import numpy as np
import pandas as pd
import torch
import os
import joblib
import io
import sys
from importlib import reload
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from copy import deepcopy

# Disable TQDM for testing
from isaac import constants
constants.TQDM_DISABLE = True

from isaac import training
from isaac.models import initialise_model, ComplexRNNModel, MultiBranchModel
from isaac.dataset import normalise, prepare_dataset


class TestEvaluate(unittest.TestCase):
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
    def is_accuracy_well_calculated(accuracy, val_loader, predicted):
        classes = np.concatenate([y.numpy() for _, y in val_loader])

        correct = 0
        for y_pred, y_true in zip(predicted, classes):
            correct += int(y_pred == y_true)
        calculated_accuracy = 100 * correct / float(len(classes))

        return np.isclose(accuracy, calculated_accuracy)

    @staticmethod
    def is_accuracy_well_calculated_for_multibranch(accuracy, val_loader, predicted, branch_id):
        classes = np.concatenate([y.numpy()[:, branch_id] for _, y in val_loader])

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

    def test_accuracy_is_correct_for_multibranch_dataset(self):
        n_features = 10
        n_classes = 3
        n_examples = 10
        seq_start, seq_end, step_size = None, None, None

        val_loader = self.create_loader(n_features, n_classes, n_examples, multibranch=True)
        model = self.create_multibranch_model(n_features, n_classes)

        return_predicted = True
        accuracy, predicted = training.evaluate(model, val_loader, return_predicted,
                                                seq_start, seq_end, step_size)

        # 2 prediction lists, one per branch
        self.assertTupleEqual(tuple(predicted.shape), (n_examples, 2))
        # 2 accuracies, one per branch
        self.assertTrue(self.is_accuracy_well_calculated_for_multibranch(accuracy[0], val_loader,
                                                                         predicted[:, 0], 0))
        self.assertTrue(self.is_accuracy_well_calculated_for_multibranch(accuracy[1], val_loader,
                                                                         predicted[:, 1], 1))

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


class TestEvaluateSavedModel(unittest.TestCase):
    model_path = "temporary_model"
    dataset_path = "temporary_dataset.h5"
    scaler_path = "temporary_scaler"

    n_trials = 10
    n_features = 5
    training_columns = ["feature_" + str(i) for i in range(n_features)]
    n_classes = 3
    class_columns = ["class_" + str(i) for i in range(n_classes)]

    hidden_dim = 5
    n_layers = 1
    network_params = (n_features, hidden_dim, n_layers, n_classes)

    def tearDown(self):
        for temp_file in [self.model_path, self.dataset_path, self.scaler_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    @staticmethod
    def write_dataframes_to_file(dataframes, path):
        for i, dataframe in enumerate(dataframes):
            dataframe.to_hdf(path, key="key_"+str(i))

    @staticmethod
    def create_dataset(n_trials, training_cols, class_cols, multiclass=False, seed=0):
        np.random.seed(seed)

        dfs = []
        seq_length = np.random.randint(25, 100)

        for _ in range(n_trials):

            if not multiclass:
                data = np.random.rand(seq_length, len(training_cols) + len(class_cols))
                classes = np.random.rand(len(class_cols))
                classes = (classes == max(classes))

                df = pd.DataFrame(data, columns=training_cols + class_cols)
                df[class_cols] = classes

            if multiclass:
                all_class_cols = list(np.concatenate(class_cols))
                data = np.random.rand(seq_length, len(training_cols) + len(all_class_cols))
                df = pd.DataFrame(data, columns=training_cols + all_class_cols)

                for class_i_cols in class_cols:
                    classes = np.random.rand(len(class_i_cols))
                    classes = (classes == max(classes))
                    df[class_i_cols] = classes

            dfs.append(df)

        return dfs

    @staticmethod
    def create_and_save_model(network_params, model_path, arch=ComplexRNNModel):
        model, _, _ = initialise_model(network_params, arch=arch)
        torch.save(model.state_dict(), model_path)

    def test_existing_dataset_path(self):
        self.create_and_save_model(self.network_params, self.model_path)
        dataset = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)
        self.write_dataframes_to_file(dataset, self.dataset_path)

        training.evaluate_saved_model(self.model_path, self.network_params, self.dataset_path,
                                      self.training_columns, self.class_columns, trials=None)

    def test_inexisting_dataset_path(self):
        self.create_and_save_model(self.network_params, self.model_path)

        with self.assertRaises(OSError):
            training.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                          self.training_columns, self.class_columns)

    def test_inexisting_path_and_trials_not_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        training.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                      self.training_columns, self.class_columns, trials=trials)

    def test_scaler_path_is_not_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        X = np.array([np.array(trial[self.training_columns]) for trial in trials])
        X = X.reshape(-1, self.n_features)
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, self.scaler_path)

        training.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                      self.training_columns, self.class_columns, trials=trials,
                                      scaler_path=self.scaler_path)

    def test_unfitted_scaler(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        scaler = StandardScaler()
        joblib.dump(scaler, self.scaler_path)

        with self.assertRaises(NotFittedError):
            training.evaluate_saved_model(self.model_path, self.network_params, None,
                                          self.training_columns, self.class_columns, trials=trials,
                                          scaler_path=self.scaler_path)

    def test_incorrect_architecture(self):
        class AlternativeArch(torch.nn.Module):
            def __init__(self, cell_type):
                super(AlternativeArch, self).__init__()
                self.rec_layer = cell_type(2, 2, 2, False)
            def forward(self, x):
                return self.rec_layer(x)[0]

        self.create_and_save_model((), self.model_path, arch=AlternativeArch)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        with self.assertRaises(RuntimeError):
            training.evaluate_saved_model(self.model_path, self.network_params, None,
                                          self.training_columns, self.class_columns, trials=trials)

    def test_incorrect_network_params(self):
        wrong_params = (4, 10, 2, 3)
        self.create_and_save_model(self.network_params, self.model_path)
        dataset = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)
        self.write_dataframes_to_file(dataset, self.dataset_path)

        with self.assertRaises(RuntimeError):
            training.evaluate_saved_model(self.model_path, wrong_params, self.dataset_path,
                                          self.training_columns, self.class_columns, trials=None)




class TestTrainingLoop(unittest.TestCase):
    def tearDown(self):
        sys.stderr = sys.__stderr__

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

    @unittest.mock.patch('sys.stderr', new_callable=io.StringIO)
    def test_print_stats_per_epoch_is_true(self, mock_stderr):
        # To test if a method prints stats in TQDM re-enable TQDM printing
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
