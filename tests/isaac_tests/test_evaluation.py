import unittest
import numpy as np
import pandas as pd
import torch
import os
import joblib

from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from isaac import evaluation
from isaac.models import initialise_model, ComplexRNNModel, MultiBranchModel


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
        accuracy, predicted = evaluation.evaluate(model, val_loader, return_predicted,
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
        accuracy, predicted = evaluation.evaluate(model, val_loader, return_predicted,
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
        accuracy = evaluation.evaluate(model, val_loader, return_predicted,
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

        evaluation.evaluate(model, val_loader, return_predicted,
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
        evaluation.evaluate(model, val_loader, return_predicted,
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
        evaluation.evaluate(model, val_loader, return_predicted,
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
            evaluation.evaluate(model, val_loader, return_predicted,
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
        evaluation.evaluate(model, val_loader, return_predicted,
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
            evaluation.evaluate(model, val_loader, return_predicted,
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
        evaluation.evaluate(model, val_loader, return_predicted,
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

        evaluation.evaluate_saved_model(self.model_path, self.network_params, self.dataset_path,
                                      self.training_columns, self.class_columns, trials=None)

    def test_inexisting_dataset_path(self):
        self.create_and_save_model(self.network_params, self.model_path)

        with self.assertRaises(OSError):
            evaluation.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                          self.training_columns, self.class_columns)

    def test_inexisting_path_and_trials_not_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        evaluation.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                      self.training_columns, self.class_columns, trials=trials)

    def test_scaler_path_is_not_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        X = np.array([np.array(trial[self.training_columns]) for trial in trials])
        X = X.reshape(-1, self.n_features)
        scaler = StandardScaler()
        scaler.fit(X)
        joblib.dump(scaler, self.scaler_path)

        evaluation.evaluate_saved_model(self.model_path, self.network_params, "invented_path.h5",
                                      self.training_columns, self.class_columns, trials=trials,
                                      scaler_path=self.scaler_path)

    def test_unfitted_scaler(self):
        self.create_and_save_model(self.network_params, self.model_path)
        trials = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)

        scaler = StandardScaler()
        joblib.dump(scaler, self.scaler_path)

        with self.assertRaises(NotFittedError):
            evaluation.evaluate_saved_model(self.model_path, self.network_params, None,
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
            evaluation.evaluate_saved_model(self.model_path, self.network_params, None,
                                          self.training_columns, self.class_columns, trials=trials)

    def test_incorrect_network_params(self):
        wrong_params = (4, 10, 2, 3)
        self.create_and_save_model(self.network_params, self.model_path)
        dataset = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)
        self.write_dataframes_to_file(dataset, self.dataset_path)

        with self.assertRaises(RuntimeError):
            evaluation.evaluate_saved_model(self.model_path, wrong_params, self.dataset_path,
                                          self.training_columns, self.class_columns, trials=None)


class TestGetBestModelAndItsAccuracy(unittest.TestCase):
    def test_model_A_is_better(self):
        model_A = "first_model"
        model_A_accuracy = 1.
        model_B = "second_model"
        model_B_accuracy = 0.5

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_A)
        self.assertEqual(best_model_accuracy, model_A_accuracy)

    def test_model_B_is_better(self):
        model_A = "first_model"
        model_A_accuracy = 0.25
        model_B = "second_model"
        model_B_accuracy = 0.6

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_B)
        self.assertEqual(best_model_accuracy, model_B_accuracy)

    def test_models_are_equally_accurate(self):
        model_A = "first_model"
        model_A_accuracy = 0.4
        model_B = "second_model"
        model_B_accuracy = 0.4

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_B)
        self.assertEqual(best_model_accuracy, model_B_accuracy)

    def test_model_returned_is_a_copy(self):
        class FakeModel:
            def __init__(self, attr):
                self.attribute = attr

        model_A = FakeModel("a")
        model_A_accuracy = 0.5
        model_B = FakeModel("b")
        model_B_accuracy = 0.55

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model.attribute, model_B.attribute)
        self.assertEqual(best_model_accuracy, model_B_accuracy)
        self.assertFalse(best_model is model_B)

class TestGetBestModelAndItsAccuracy(unittest.TestCase):
    def test_model_A_is_better(self):
        model_A = "first_model"
        model_A_accuracy = 1.
        model_B = "second_model"
        model_B_accuracy = 0.5

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_A)
        self.assertEqual(best_model_accuracy, model_A_accuracy)

    def test_model_B_is_better(self):
        model_A = "first_model"
        model_A_accuracy = 0.25
        model_B = "second_model"
        model_B_accuracy = 0.6

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_B)
        self.assertEqual(best_model_accuracy, model_B_accuracy)

    def test_models_are_equally_accurate(self):
        model_A = "first_model"
        model_A_accuracy = 0.4
        model_B = "second_model"
        model_B_accuracy = 0.4

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model, model_B)
        self.assertEqual(best_model_accuracy, model_B_accuracy)

    def test_model_returned_is_a_copy(self):
        class FakeModel:
            def __init__(self, attr):
                self.attribute = attr

        model_A = FakeModel("a")
        model_A_accuracy = 0.5
        model_B = FakeModel("b")
        model_B_accuracy = 0.55

        best_model, best_model_accuracy = evaluation.get_best_model_and_its_accuracy(model_A,
                                                                                   model_A_accuracy,
                                                                                   model_B,
                                                                                   model_B_accuracy)

        self.assertEqual(best_model.attribute, model_B.attribute)
        self.assertEqual(best_model_accuracy, model_B_accuracy)
        self.assertFalse(best_model is model_B)


if __name__ == "__main__":
    unittest.main()
