import unittest
import pandas as pd
import numpy as np
import os

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

if __name__ == "__main__":
    unittest.main()