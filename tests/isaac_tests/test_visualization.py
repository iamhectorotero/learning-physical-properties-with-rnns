import unittest
import os
import torch
import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use('TkAgg')
# Using the default backend results in segfault

from matplotlib.axes import Axes

from isaac import training, visualization
from isaac.dataset import prepare_dataset

from isaac.models import ComplexRNNModel, initialise_model

# Disable TQDM for testing
from isaac import constants
constants.TQDM_DISABLE = True


class TestPlotConfusionMatrixGivenPredictedAndTestLoader(unittest.TestCase):
    model_path = "temporary_model"
    dataset_path = "temporary_dataset.h5"
    scaler_path = "temporary_scaler"
    plot_path = "temporary_plot.pdf"

    n_trials = 10
    n_features = 5
    training_columns = ["feature_" + str(i) for i in range(n_features)]
    n_classes = 3
    class_columns = ["class_" + str(i) for i in range(n_classes)]

    hidden_dim = 5
    n_layers = 1
    network_params = (n_features, hidden_dim, n_layers, n_classes)

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

    def tearDown(self):
        for temp_file in [self.model_path, self.dataset_path, self.scaler_path, self.plot_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def test_save_plot_is_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        dataset = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)
        test_loader, _ = prepare_dataset([dataset], class_columns=self.class_columns,
                                         training_columns=self.training_columns)

        self.write_dataframes_to_file(dataset, self.dataset_path)
        _, predicted = training.evaluate_saved_model(self.model_path, self.network_params,
                                                     self.dataset_path, self.training_columns,
                                                     self.class_columns, trials=None)

        plot_path = None
        ax = visualization.plot_confusion_matrix_given_predicted_and_test_loader(predicted, test_loader,
                                                                            self.class_columns,
                                                                            plot_path)
        self.assertTrue(isinstance(ax, Axes))
        self.assertFalse(os.path.exists(self.plot_path))

    def test_save_plot_is_not_none(self):
        self.create_and_save_model(self.network_params, self.model_path)
        dataset = self.create_dataset(self.n_trials, self.training_columns, self.class_columns)
        test_loader, _ = prepare_dataset([dataset], class_columns=self.class_columns,
                                         training_columns=self.training_columns)

        self.write_dataframes_to_file(dataset, self.dataset_path)
        _, predicted = training.evaluate_saved_model(self.model_path, self.network_params,
                                                     self.dataset_path, self.training_columns,
                                                     self.class_columns, trials=None)

        ax = visualization.plot_confusion_matrix_given_predicted_and_test_loader(predicted, test_loader,
                                                                            self.class_columns,
                                                                            self.plot_path)
        self.assertTrue(isinstance(ax, Axes))
        self.assertTrue(os.path.exists(self.plot_path))


if __name__ == "__main__":
    unittest.main()
