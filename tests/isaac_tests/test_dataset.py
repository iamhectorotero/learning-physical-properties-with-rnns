import unittest
import pandas as pd
import numpy as np
import os
import torch
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

from functools import reduce
from isaac import dataset
from isaac.constants import MASS_CLASS_COLS, FORCE_CLASS_COLS


class TestReadDataset(unittest.TestCase):
    hdf_path = "temporary_hdf.h5"

    @staticmethod
    def write_dataframes_to_file(dataframes, path):
        for i, dataframe in enumerate(dataframes):
            dataframe.to_hdf(path, key="key_"+str(i))

    @staticmethod
    def are_columns_in_all_trials(dataframes, columns):
        columns = set(columns)
        for df in dataframes:
            if not columns.issubset(df.columns):
                return False

        return True

    @staticmethod
    def check_columns_are_not_in_all_trials(dataframes, columns):
        for df in dataframes:
            for column in columns:
                if column in df.columns:
                    return False
        return True

    def tearDown(self):
        if os.path.exists(self.hdf_path):
            os.remove(self.hdf_path)

    def test_non_existing_path(self):
        self.assertRaises(OSError, dataset.read_dataset, "invented_path.h5")

    def test_n_trials_is_none(self):
        n_trials = 5
        dfs = [pd.DataFrame(np.random.rand(10, 4)) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path)
        self.assertEqual(n_trials, len(read_dataset))

    def test_n_trials_larger_than_available(self):
        n_trials = 5
        dfs = [pd.DataFrame(np.random.rand(10, 4)) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, n_trials=n_trials*5)
        self.assertEqual(n_trials, len(read_dataset))

    def test_n_trials_smaller_than_available(self):
        n_trials = 5
        dfs = [pd.DataFrame(np.random.rand(10, 4)) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        trials_to_read = 2
        read_dataset = dataset.read_dataset(self.hdf_path, n_trials=trials_to_read)
        self.assertEqual(trials_to_read, len(read_dataset))

    def test_cols_that_do_not_exist(self):
        n_trials = 5
        dfs = [pd.DataFrame(np.random.rand(10, 4)) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        inexisting_cols = ["inexisting_col_a", "inexisting_col_b"]
        with self.assertRaises(KeyError):
            dataset.read_dataset(self.hdf_path, cols=inexisting_cols, add_class_columns=False)

    def test_subset_of_cols(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]
        dfs = [pd.DataFrame(np.random.rand(10, 4), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, cols=cols[:2], add_class_columns=False)
        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols[:2]))
        self.assertTrue(self.check_columns_are_not_in_all_trials(read_dataset, cols[2:]))

    def test_duplicated_cols_are_only_read_once(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]
        dfs = [pd.DataFrame(np.random.rand(10, 4), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, cols=cols+cols, add_class_columns=False)
        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols))
        self.assertTrue(reduce(lambda x,y: x and y, [len(df.columns) == 4 for df in read_dataset]))

    def test_cols_is_none(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]
        dfs = [pd.DataFrame(np.random.rand(10, 4), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, cols=None, add_class_columns=False)
        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols))

    def test_add_class_columns(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]
        cols_to_write = cols + list(MASS_CLASS_COLS) + list(FORCE_CLASS_COLS)

        dfs = [pd.DataFrame(np.random.rand(10, len(cols_to_write)), columns=cols_to_write) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, cols=cols, add_class_columns=True)
        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols_to_write))

    def test_do_not_add_class_columns(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]
        cols_to_write = cols + list(MASS_CLASS_COLS) + list(FORCE_CLASS_COLS)

        dfs = [pd.DataFrame(np.random.rand(10, len(cols_to_write)), columns=cols_to_write) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        read_dataset = dataset.read_dataset(self.hdf_path, cols=cols, add_class_columns=False)
        self.assertFalse(self.are_columns_in_all_trials(read_dataset, cols_to_write))
        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols))

    def test_do_not_add_class_columns_when_cols_unspecified(self):
        n_trials = 5
        cols = ["col_"+str(i) for i in range(4)]

        dfs = [pd.DataFrame(np.random.rand(10, len(cols)), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        try:
            read_dataset = dataset.read_dataset(self.hdf_path, cols=None, add_class_columns=True)
        except KeyError:
            self.fail("read_dataset failed unexpectedly")

        self.assertTrue(self.are_columns_in_all_trials(read_dataset, cols))

    def test_reproducibility_for_subset_of_trials(self):
        n_trials = 25
        cols = ["col_"+str(i) for i in range(4)]

        dfs = [pd.DataFrame(np.random.rand(10, len(cols)), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        first_dataset = dataset.read_dataset(self.hdf_path, n_trials=4, seed=90)
        second_dataset = dataset.read_dataset(self.hdf_path, n_trials=4, seed=90)

        for df1, df2 in zip(first_dataset, second_dataset):
            pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_for_subset_of_trials(self):
        n_trials = 25
        cols = ["col_"+str(i) for i in range(4)]

        dfs = [pd.DataFrame(np.random.rand(10, len(cols)), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        first_dataset = dataset.read_dataset(self.hdf_path, n_trials=4, seed=90)
        second_dataset = dataset.read_dataset(self.hdf_path, n_trials=4, seed=0)

        different_order = False
        for df1, df2 in zip(first_dataset, second_dataset):
            if not df1.equals(df2):
                different_order = True
                break

        self.assertTrue(different_order)

    def test_different_seeds_for_reading_all_trials(self):
        n_trials = 25
        cols = ["col_"+str(i) for i in range(4)]

        dfs = [pd.DataFrame(np.random.rand(10, len(cols)), columns=cols) for _ in range(n_trials)]
        self.write_dataframes_to_file(dfs, self.hdf_path)

        first_dataset = dataset.read_dataset(self.hdf_path, seed=90)
        second_dataset = dataset.read_dataset(self.hdf_path, seed=0)

        different_order = False
        for df1, df2 in zip(first_dataset, second_dataset):
            if not df1.equals(df2):
                different_order = True
                break

        self.assertTrue(different_order)


class TestNormalise(unittest.TestCase):
    @staticmethod
    def are_mean_and_variance_correct(X, atol=1e-8):
        X = X.reshape(-1, X.shape[-1])
        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0)
        return (np.isclose(means, np.zeros_like(means), atol=atol).all() and
                np.isclose(variances, np.ones_like(variances), atol=atol).all())

    @staticmethod
    def is_shape_correct(original_shape, X):
        return original_shape == X.shape

    def test_fitting_scaler(self):
        scaler = StandardScaler()
        original_shape = (25, 10, 7)
        X = np.random.rand(*original_shape)
        X = dataset.normalise(X, scaler, fit_scaler=True)

        self.assertTrue(self.is_shape_correct(original_shape, X))
        self.assertTrue(self.are_mean_and_variance_correct(X))

    def test_using_previously_fitted_scaler(self):
        scaler = StandardScaler()
        original_shape = (25, 10, 7)
        X = np.random.rand(*original_shape)
        X = dataset.normalise(X, scaler, fit_scaler=True)

        scale_ = scaler.scale_
        mean_= scaler.mean_
        n_samples_seen_ = scaler.n_samples_seen_

        X2 = np.random.rand(50, 4, 7)
        X2 = dataset.normalise(X, scaler, fit_scaler=False)

        self.assertTrue(np.isclose(scaler.scale_, scale_).all())
        self.assertTrue(np.isclose(scaler.mean_, mean_).all())
        self.assertEqual(scaler.n_samples_seen_, n_samples_seen_)

    def test_using_previously_not_fitted_scaler(self):
        scaler = StandardScaler()
        original_shape = (25, 10, 7)
        X = np.random.rand(*original_shape)

        with self.assertRaises(NotFittedError):
            dataset.normalise(X, scaler, fit_scaler=False)

    def test_not_normalising_some_columns(self):
        scaler = StandardScaler()
        original_shape = (25, 10, 7)
        columns_to_normalise_bool_index = np.array([True, True, False, True, False, True, True])

        original_X = np.random.rand(*original_shape)
        X = dataset.normalise(original_X, scaler, True, columns_to_normalise_bool_index)

        self.assertTrue(np.isclose(original_X[:, :, columns_to_normalise_bool_index],
                                   X[:, :, columns_to_normalise_bool_index]).all())

        self.assertTrue(self.are_mean_and_variance_correct(X[:, :, columns_to_normalise_bool_index]))

    def test_not_normalising_all_columns(self):
        scaler = StandardScaler()
        original_shape = (25, 10, 7)
        columns_to_normalise_bool_index = np.array([False for _ in range(original_shape[-1])])

        X = np.random.rand(*original_shape)
        with self.assertRaises(ValueError):
            dataset.normalise(X, scaler, True, columns_to_normalise_bool_index)


class TestPrepareDataset(unittest.TestCase):
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
    def are_trials_correctly_set(dataset_loader, dataset, training_cols=None, class_cols=None):
        for batch_x, batch_y in dataset_loader:
            for loader_x, loader_y, trial in zip(batch_x, batch_y, dataset):
                if training_cols is not None:
                    if not np.isclose(loader_x.cpu().numpy(), np.array(trial[training_cols])).all():
                        return False

                if class_cols is not None:
                    class_idx = np.argmax(np.array(trial[class_cols].iloc[0]))
                    if not loader_y.cpu().numpy() == class_idx:
                        return False

        return True

    @staticmethod
    def is_multiclass_correctly_set(dataset_loader, dataset, multiclass_list):
        for batch_x, batch_y in dataset_loader:
            for loader_x, loader_y, trial in zip(batch_x, batch_y, dataset):
                for class_values, class_i_cols in zip(loader_y, multiclass_list):
                    class_idx = np.argmax(np.array(trial[class_i_cols].iloc[0]))
                    if not class_values.cpu().numpy() == class_idx:
                        return False
        return True

    def test_one_dataset(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        dataset_loader, scaler = dataset.prepare_dataset([test_dataset], class_cols, training_columns=training_cols)

        self.assertEqual(scaler, None)
        self.assertEqual(len(dataset_loader.dataset), n_trials)
        self.assertTrue(self.are_trials_correctly_set(dataset_loader, test_dataset, training_cols,
                                                      class_cols))

    def test_more_than_one_dataset(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        train_dataset = self.create_dataset(2*n_trials, training_cols, class_cols)
        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        loaders, scaler = dataset.prepare_dataset([train_dataset, test_dataset], class_cols,
                                                  training_columns=training_cols)
        self.assertEqual(scaler, None)
        self.assertEqual(len(loaders[0].dataset), 2*n_trials)
        self.assertEqual(len(loaders[1].dataset), n_trials)
        self.assertTrue(self.are_trials_correctly_set(loaders[0], train_dataset, training_cols,
                                                      class_cols))
        self.assertTrue(self.are_trials_correctly_set(loaders[1], test_dataset, training_cols,
                                                      class_cols))

    def test_non_categorical_class_columns(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        for trial in test_dataset:
            trial[class_cols] = np.random.rand(*trial[class_cols].shape)

        with self.assertRaisesRegex(ValueError, "Classes are not one-hot encoded"):
            dataset.prepare_dataset([test_dataset], class_cols, training_columns=training_cols)

    def test_non_categorical_class_columnns_in_multiclass_setting(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 6
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]
        multiclass_list = [class_cols[:3], class_cols[3:]]

        test_dataset = self.create_dataset(n_trials, training_cols, multiclass_list, multiclass=True)
        for trial in test_dataset:
            trial[class_cols] = np.random.rand(*trial[class_cols].shape)

        with self.assertRaises(ValueError):
            dataset.prepare_dataset([test_dataset], multiclass_list, multiclass=True,
                                    training_columns=training_cols)

    def test_class_columnns_not_in_every_dataset(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 5
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        train_dataset = self.create_dataset(2*n_trials, training_cols, class_cols)
        test_dataset = self.create_dataset(n_trials, training_cols, class_cols[:3])

        with self.assertRaises(KeyError):
            dataset.prepare_dataset([train_dataset, test_dataset], class_cols,
                                    training_columns=training_cols)

    def test_class_columnns_integer(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)

        for trial in test_dataset:
            trial[class_cols] = trial[class_cols]

        dataset_loader, _ = dataset.prepare_dataset([test_dataset], class_cols, training_columns=training_cols)

        self.assertTrue(self.are_trials_correctly_set(dataset_loader, test_dataset, training_cols,
                                                      class_cols))

    def test_multiclass(self):
        n_trials = 10
        n_training_cols = 8
        n_class_cols = 6
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]
        multiclass_list = [class_cols[:3], class_cols[3:]]

        test_dataset = self.create_dataset(n_trials, training_cols, multiclass_list, multiclass=True)
        dataset_loader, scaler = dataset.prepare_dataset([test_dataset], multiclass_list, multiclass=True,
                                                        training_columns=training_cols)

        self.assertEqual(scaler, None)
        self.assertEqual(len(dataset_loader.dataset), n_trials)

        # Check only if training cols are correctly set by passing None
        self.assertTrue(self.are_trials_correctly_set(dataset_loader, test_dataset, training_cols,
                                                      None))
        self.assertTrue(self.is_multiclass_correctly_set(dataset_loader, test_dataset,
                                                         multiclass_list))

    def test_different_batch_sizes(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        for batch_size in [1, 4, 16]:
            dataset_loader, scaler = dataset.prepare_dataset([test_dataset], class_cols,
                                                             training_columns=training_cols,
                                                             batch_size=batch_size)

            self.assertEqual(dataset_loader.batch_size, batch_size)

    def test_fitting_scaler(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        dataset_loader, scaler = dataset.prepare_dataset([test_dataset], class_cols,
                                                         training_columns=training_cols,
                                                         normalise_data=True, scaler=None)

        self.assertTrue(scaler is not None)
        numpy_dataset = dataset_loader.dataset.tensors[0].numpy()
        self.assertTrue(TestNormalise.are_mean_and_variance_correct(numpy_dataset, atol=1e-7))

    def test_not_fitting_scaler_but_normalising_data(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols, seed=0)
        second_dataset = self.create_dataset(n_trials, training_cols, class_cols, seed=71)
        _, scaler = dataset.prepare_dataset([test_dataset], class_cols,
                                            training_columns=training_cols,
                                            normalise_data=True, scaler=None)

        self.assertTrue(scaler is not None)
        loaders, scaler = dataset.prepare_dataset([second_dataset, test_dataset], class_cols,
                                                  training_columns=training_cols,
                                                  normalise_data=True, scaler=scaler)

        self.assertTrue(scaler is not None)
        numpy_dataset = loaders[1].dataset.tensors[0].numpy()
        self.assertTrue(TestNormalise.are_mean_and_variance_correct(numpy_dataset, atol=1e-7))

    def test_passing_scaler_but_not_normalising_data(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols, seed=0)
        second_dataset = self.create_dataset(n_trials, training_cols, class_cols, seed=71)
        _, scaler = dataset.prepare_dataset([test_dataset], class_cols,
                                            training_columns=training_cols,
                                            normalise_data=True, scaler=None)

        self.assertTrue(scaler is not None)
        loader, scaler = dataset.prepare_dataset([second_dataset], class_cols,
                                                  training_columns=training_cols,
                                                  normalise_data=False, scaler=scaler)

        self.assertTrue(scaler is not None)

        numpy_dataset = loader.dataset.tensors[0].numpy()
        datasets_training_cols = np.array([np.array(trial[training_cols]) for trial in second_dataset])
        self.assertTrue(np.isclose(numpy_dataset, datasets_training_cols).all())

    def test_transforms(self):
        def add_one(dataframe):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                dataframe.loc[:, dataframe.columns] = dataframe[dataframe.columns] + 1

        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        loader, _ = dataset.prepare_dataset([test_dataset], class_cols,
                                             training_columns=training_cols,
                                             normalise_data=False, transforms=[add_one])

        numpy_dataset = loader.dataset.tensors[0].numpy()
        datasets_training_cols = np.array([np.array(trial[training_cols]) for trial in test_dataset])
        self.assertTrue(np.isclose(numpy_dataset, datasets_training_cols + 1).all())

    def test_categorical_columns(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        categorical_columns = training_cols[:2]
        non_categorical_columns = training_cols[2:]
        categorical_cols_bool_index = np.array([col in categorical_columns for col in training_cols])

        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        loader, _ = dataset.prepare_dataset([test_dataset], class_cols,
                                             training_columns=training_cols,
                                             normalise_data=True,
                                             categorical_columns=categorical_columns)

        numpy_dataset = loader.dataset.tensors[0].numpy()
        datasets_training_cols = np.array([np.array(trial[training_cols])
                                           for trial in test_dataset])

        # Check categorical columns haven't changed
        self.assertTrue(np.isclose(numpy_dataset[:, :, categorical_cols_bool_index],
                                   datasets_training_cols[:, :, categorical_cols_bool_index]).all())

        # Check non-categorical columns have been normalised correctly
        self.assertTrue(TestNormalise.are_mean_and_variance_correct(numpy_dataset[:, :, ~categorical_cols_bool_index], atol=1e-7))

    def test_all_training_columns_are_categorical(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        categorical_columns = training_cols

        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)

        with self.assertRaises(ValueError):
            dataset.prepare_dataset([test_dataset], class_cols,
                                    training_columns=training_cols,
                                    normalise_data=True,
                                    categorical_columns=categorical_columns)

    def test_normalisation_cols(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        categorical_columns = training_cols[:2]
        non_categorical_columns = training_cols[2:]
        categorical_cols_bool_index = np.array([col in categorical_columns for col in training_cols])

        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        loader, _ = dataset.prepare_dataset([test_dataset], class_cols,
                                             training_columns=training_cols,
                                             normalise_data=True,
                                             normalisation_cols=non_categorical_columns)

        numpy_dataset = loader.dataset.tensors[0].numpy()
        datasets_training_cols = np.array([np.array(trial[training_cols])
                                           for trial in test_dataset])

        # Check categorical columns haven't changed
        self.assertTrue(np.isclose(numpy_dataset[:, :, categorical_cols_bool_index],
                                   datasets_training_cols[:, :, categorical_cols_bool_index]).all())

        # Check non-categorical columns have been normalised correctly
        self.assertTrue(TestNormalise.are_mean_and_variance_correct(numpy_dataset[:, :, ~categorical_cols_bool_index], atol=1e-7))

    def test_passing_both_categorical_and_normalisation_cols(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        categorical_columns = training_cols[:2]
        non_categorical_columns = training_cols[2:4]
        columns_not_specified = training_cols[4:6]
        non_categorical_cols_bool_index = np.array([col in non_categorical_columns for col in training_cols])

        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        loader, _ = dataset.prepare_dataset([test_dataset], class_cols,
                                             training_columns=training_cols,
                                             normalise_data=True,
                                             normalisation_cols=non_categorical_columns,
                                             categorical_columns=categorical_columns)

        numpy_dataset = loader.dataset.tensors[0].numpy()
        datasets_training_cols = np.array([np.array(trial[training_cols])
                                           for trial in test_dataset])

        # Check categorical columns haven't changed
        self.assertTrue(np.isclose(numpy_dataset[:, :, ~non_categorical_cols_bool_index],
                                   datasets_training_cols[:, :, ~non_categorical_cols_bool_index]).all())

        # Check non-categorical columns have been normalised correctly
        self.assertTrue(TestNormalise.are_mean_and_variance_correct(numpy_dataset[:, :, non_categorical_cols_bool_index], atol=1e-7))

    def test_columns_in_both_categorical_and_normalisation_cols(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        categorical_columns = training_cols[:4]
        non_categorical_columns = training_cols[2:]

        class_cols = ["class_"+str(i) for i in range(n_class_cols)]

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)

        with self.assertRaises(ValueError):
            dataset.prepare_dataset([test_dataset], class_cols,
                                    training_columns=training_cols,
                                    normalise_data=True,
                                    normalisation_cols=non_categorical_columns,
                                    categorical_columns=categorical_columns)

    def test_gpu_device(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]
        gpu_device = torch.device("cuda:0")

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        dataset_loader, _= dataset.prepare_dataset([test_dataset], class_cols, 
                                                   training_columns=training_cols,
                                                   device=gpu_device)

        if torch.cuda.is_available():
            self.assertTrue(dataset_loader.dataset.tensors[0].is_cuda)
            self.assertTrue(dataset_loader.dataset.tensors[1].is_cuda)
        else:
            warnings.warn("No Cuda device available to run this test.")

    def test_cpu_device(self):
        n_trials = 10
        n_training_cols = 6
        n_class_cols = 3
        training_cols = ["col_"+str(i) for i in range(n_training_cols)]
        class_cols = ["class_"+str(i) for i in range(n_class_cols)]
        cpu_device = torch.device("cpu")

        test_dataset = self.create_dataset(n_trials, training_cols, class_cols)
        dataset_loader, _= dataset.prepare_dataset([test_dataset], class_cols, 
                                                   training_columns=training_cols,)

        self.assertFalse(dataset_loader.dataset.tensors[0].is_cuda)
        self.assertFalse(dataset_loader.dataset.tensors[1].is_cuda)


class TestAreClassesOneHotEncoded(unittest.TestCase):
    def test_boolean_classes(self):
        classes = [[False, False, True], [True, False, False], [False, True, False]]
        for class_option in classes:
            self.assertTrue(dataset.are_classes_one_hot_encoded(class_option))

    def test_more_than_three_classes(self):
        classes = [False, False, True, False, False, False, False]
        self.assertTrue(dataset.are_classes_one_hot_encoded(classes))

    def test_integer_classes(self):
        classes = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]

        for class_option in classes:
            self.assertTrue(dataset.are_classes_one_hot_encoded(class_option))

    def test_not_one_hot_encoded_classes(self):
        classes = [[1, 0, 1], [False, True, True], [True, 1, False]]

        for class_option in classes:
            self.assertFalse(dataset.are_classes_one_hot_encoded(class_option))

    def test_non_binary_classes(self):
        classes = [[1.23, 0.3, 1.23], ["", "c", ""]]

        for class_option in classes:
            self.assertFalse(dataset.are_classes_one_hot_encoded(class_option))


if __name__ == "__main__":
    unittest.main()
