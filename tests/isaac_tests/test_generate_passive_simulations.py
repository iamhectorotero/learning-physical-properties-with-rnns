import unittest
import numpy as np
import os
import pandas as pd

# Disable TQDM for testing
from isaac import constants
constants.TQDM_DISABLE = True

from isaac import generate_passive_simulations as gps
from isaac.constants import BASIC_TRAINING_COLS, MASS_CLASS_COLS, FORCE_CLASS_COLS
from isaac.dataset import read_dataset
from simulator.config import TIMEOUT, generate_cond

class TestGetMassAnswer(unittest.TestCase):
    def test_a_is_larger(self):
        masses = [10, 8, 12, 90]
        self.assertEqual(gps.get_mass_answer(masses), 'A')

    def test_b_is_larger(self):
        masses = [6, 7, 5]
        self.assertEqual(gps.get_mass_answer(masses), 'B')

    def test_same(self):
        masses = [5, 5]
        self.assertEqual(gps.get_mass_answer(masses), 'same')

    def test_list_too_short(self):
        masses = [2]
        with self.assertRaises(IndexError):
            gps.get_mass_answer(masses)

    def test_empty_list(self):
        masses = []
        with self.assertRaises(IndexError):
            gps.get_mass_answer(masses)


class TestGetForceAnswer(unittest.TestCase):
    def test_attract(self):
        forces = np.random.rand(2, 2)
        self.assertEqual(gps.get_force_answer(forces), 'attract')

    def test_repel(self):
        forces = np.random.rand(3, 3)
        forces[0][1] *= -1
        self.assertEqual(gps.get_force_answer(forces), 'repel')

    def test_same(self):
        forces = np.zeros((4, 4))
        self.assertEqual(gps.get_force_answer(forces), 'none')

    def test_list_too_short(self):
        forces = [[0]]
        with self.assertRaises(IndexError):
            gps.get_force_answer(forces)

    def test_empty_list(self):
        forces = []
        with self.assertRaises(IndexError):
            gps.get_force_answer(forces)


class TestGetForceAnswerFromFlatList(unittest.TestCase):
    def test_attract(self):
        forces = [1]
        self.assertEqual(gps.get_force_answer_from_flat_list(forces), 'attract')

    def test_repel(self):
        forces = [-1]
        self.assertEqual(gps.get_force_answer_from_flat_list(forces), 'repel')

    def test_same(self):
        forces = [0]
        self.assertEqual(gps.get_force_answer_from_flat_list(forces), 'none')

    def test_empty_list(self):
        forces = []
        with self.assertRaises(IndexError):
            gps.get_force_answer_from_flat_list(forces)


class TestGetConfigurationAnswer(unittest.TestCase):
    def test_attract(self):
        config = [[0, 0], [1]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[1], 'attract')

    def test_repel(self):
        config = [[0, 0], [-1]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[1], 'repel')

    def test_same(self):
        config = [[0, 0], [0]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[1], 'none')

    def test_a_is_larger(self):
        config = [[1, 0], [0]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[0], 'A')

    def test_b_is_larger(self):
        config = [[0, 1], [0]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[0], 'B')

    def test_same(self):
        config = [[0, 0], [0]]
        answer = gps.get_configuration_answer(config)
        self.assertEqual(answer.split("_")[0], 'same')


class TestCartesianProduct(unittest.TestCase):
    @staticmethod
    def is_size_correct(a, b, cp):
        return len(a)*len(b) == len(cp)

    @staticmethod
    def are_components_correctly_set(a, b, cp):
        idx = 0
        for i in a:
            for j in b:
                if not cp[idx] == (i, j):
                   return False
                idx += 1
        return True

    def test_same_size_lists(self):
        a = ["a", "b", "c", "d"]
        b = [1, 2, 3, 4]
        cp = gps.cartesian_product(a, b)

        self.assertTrue(self.is_size_correct(a, b, cp))
        self.assertTrue(self.are_components_correctly_set(a, b, cp))

    def test_diffferent_size_lists(self):
        a = ["a", "b", "c", "d"]
        b = [1, 2]
        cp = gps.cartesian_product(a, b)

        self.assertTrue(self.is_size_correct(a, b, cp))
        self.assertTrue(self.are_components_correctly_set(a, b, cp))

    def test_one_empty_list(self):
        a = []
        b = [1, 2, 3, 4]
        cp = gps.cartesian_product(a, b)

        self.assertTrue(self.is_size_correct(a, b, cp))

    def test_two_empty_lists(self):
        a = []
        b = []
        cp = gps.cartesian_product(a, b)

        self.assertTrue(self.is_size_correct(a, b, cp))


class TestGenerateEveryWorldConfiguration(unittest.TestCase):
    def test_only_existing_case(self):
        configurations = gps.generate_every_world_configuration()
        self.assertEqual(configurations.shape[0], 3**7)
        self.assertEqual(configurations.shape[1], 2)

        mass_conf, force_conf = configurations[:, 0], configurations[:, 1]
        mass_conf_set = set(tuple(conf) for conf in mass_conf)
        force_conf_set = set(tuple(conf) for conf in force_conf)

        self.assertEqual(len(mass_conf_set), 3)
        for conf in mass_conf_set:
            self.assertEqual(len(conf), 4)
            self.assertTrue(conf[:2] in [(1, 1), (1, 2), (2, 1)])
            self.assertEqual(conf[2:], (1, 1))

        self.assertEqual(len(force_conf_set), 3**6)
        for conf in force_conf_set:
            self.assertEqual(len(conf), 6)
            self.assertTrue(set(conf).issubset(set([0, 1, 2])))


class TestSimulateTrial(unittest.TestCase):
    @staticmethod
    def are_answers_correctly_set(trial_conf, trial_data):
        df_force_answer = np.array(trial_data[list(FORCE_CLASS_COLS)].iloc[0])
        df_string_force_answer = FORCE_CLASS_COLS[df_force_answer.argmax()]
        conf_string_force_answer = gps.get_force_answer(trial_conf[0]["lf"])

        df_mass_answer = np.array(trial_data[list(MASS_CLASS_COLS)].iloc[0])
        df_string_mass_answer = MASS_CLASS_COLS[df_mass_answer.argmax()]
        conf_string_mass_answer = gps.get_mass_answer(trial_conf[0]["mass"])

        return (conf_string_force_answer == df_string_force_answer) and (conf_string_mass_answer == df_string_mass_answer)

    def test_standard_trial(self):
        np.random.seed(0)

        every_world_configuration = gps.generate_every_world_configuration()
        trial_conf_idx = np.random.randint(0, len(every_world_configuration))
        trial_conf = generate_cond([every_world_configuration[trial_conf_idx]])
        trial_data = gps.simulate_trial(trial_conf[0])

        # TODO: make len(trial_data) actually be equal to TIMEOUT
        self.assertEqual(len(trial_data), TIMEOUT + 1)

        for col in BASIC_TRAINING_COLS:
            self.assertTrue(col in trial_data.columns)

        for answer_col in MASS_CLASS_COLS + FORCE_CLASS_COLS:
            self.assertTrue(col in trial_data.columns)

        self.assertTrue(self.are_answers_correctly_set(trial_conf, trial_data))


class TestCreatePassiveDatasetsForTraining(unittest.TestCase):
    train_hdf_path = "temporary_train_hdf.h5"
    val_hdf_path = "temporary_val_hdf.h5"
    test_hdf_path = "temporary_test_hdf.h5"

    args = {"train_hdf_path": train_hdf_path, "val_hdf_path": val_hdf_path,
            "test_hdf_path": test_hdf_path, "n_simulations_train": 3, "n_simulations_val": 1,
            "n_simulations_test": 2, "train_size": 0.33, "val_size": 0.33, "test_size": 0.33,
            "n_processes": 1, "trial_hdf_key_prefix": "trial_"}

    def tearDown(self):
        for path in [self.train_hdf_path, self.val_hdf_path, self.test_hdf_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_dataset_paths(self):
        gps.create_passive_datasets_for_training(**self.args)

        for path in [self.train_hdf_path, self.val_hdf_path, self.test_hdf_path]:
            self.assertTrue(os.path.exists(path))

    def test_datasets_sizes(self):
        gps.create_passive_datasets_for_training(**self.args)

        for path, dataset_trials in zip([self.train_hdf_path, self.val_hdf_path, self.test_hdf_path],
                                        [self.args["n_simulations_train"],
                                         self.args["n_simulations_val"],
                                         self.args["n_simulations_test"]]):
            dataset = read_dataset(path)
            self.assertEqual(len(dataset), dataset_trials)

    def test_datasets_sizes_with_previously_existing_dataset(self):
        # Create initial datasets with default key_prefix
        gps.create_passive_datasets_for_training(**self.args)

        # Create new datasets with a different key_prefix and number of trials
        # If the previous dataset isn't deleted: either there will be more trials (added to the
        # same path) or an incorrect number of trials.
        args_copy = self.args.copy()
        args_copy["trial_hdf_key_prefix"] = "newtrials_"
        args_copy["n_simulations_train"] = 1
        args_copy["n_simulations_val"] = 1
        args_copy["n_simulations_test"] = 1
        gps.create_passive_datasets_for_training(**args_copy)

        for path, dataset_trials in zip([self.train_hdf_path, self.val_hdf_path, self.test_hdf_path],
                                        [args_copy["n_simulations_train"],
                                         args_copy["n_simulations_val"],
                                         args_copy["n_simulations_test"]]):
            dataset = read_dataset(path)
            self.assertEqual(len(dataset), dataset_trials)

    def test_empty_dataset(self):
        args_copy = self.args.copy()
        args_copy["n_simulations_train"] = 0

        gps.create_passive_datasets_for_training(**args_copy)

        for path, dataset_trials in zip([self.train_hdf_path, self.val_hdf_path, self.test_hdf_path],
                                        [args_copy["n_simulations_train"],
                                         args_copy["n_simulations_val"],
                                         args_copy["n_simulations_test"]]):
            if dataset_trials > 0:
                dataset = read_dataset(path)
                self.assertEqual(len(dataset), dataset_trials)
            else:
                self.assertFalse(os.path.exists(path))

    def test_all_datasets_empty(self):
        args_copy = self.args.copy()
        args_copy["n_simulations_train"] = 0
        args_copy["n_simulations_val"] = 0
        args_copy["n_simulations_test"] = 0

        gps.create_passive_datasets_for_training(**args_copy)

        for path, dataset_trials in zip([self.train_hdf_path, self.val_hdf_path, self.test_hdf_path],
                                        [args_copy["n_simulations_train"],
                                         args_copy["n_simulations_val"],
                                         args_copy["n_simulations_test"]]):
            if dataset_trials > 0:
                dataset = read_dataset(path)
                self.assertEqual(len(dataset), dataset_trials)
            else:
                self.assertFalse(os.path.exists(path))

    def test_trial_hdf_key_prefix(self):
        gps.create_passive_datasets_for_training(**self.args)

        for path in [self.train_hdf_path, self.val_hdf_path, self.test_hdf_path]:
            try:
                pd.read_hdf(path, key=self.args["trial_hdf_key_prefix"]+str(0))
            except:
                self.fail()

if __name__ == "__main__":
    unittest.main()
