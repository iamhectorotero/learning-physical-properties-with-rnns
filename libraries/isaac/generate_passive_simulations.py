from simulator.environment import physic_env
import numpy as np
from simulator.config import *
from tqdm import tqdm
import pandas as pd
import os
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from simulator.config import generate_possible

from .constants import TQDM_DISABLE


def get_mass_answer(masses):
    """Returns a string-based answer given a configuration of masses.
    Args:
        masses: a 1D vector (len(masses) >= 2)
    Returns:
        answer: 'A' if the first mass is larger than the second, 'B' if the second is larger and
        'same' otherwise."""
    if masses[0] > masses[1]:
        return 'A'
    elif masses[1] > masses[0]:
        return 'B'
    return 'same'


def get_force_answer(forces):
    """Returns a string-based answer given a configuration of forces.
    Args:
        forces: a square matrix >= 2x2
    Returns:
        answer: 'attract' if the force between the first and second puck is positive, 'repel' if it
         is negative and 'none' otherwise."""
    if forces[0][1] > 0:
        return "attract"
    elif forces[0][1] < 0:
        return "repel"
    return "none"


def get_force_answer_from_flat_list(forces):
    """Returns a string-based answer given a configuration of forces.
    Args:
        forces: a vector corresponding to the non-diagonal values of the force configuration.
    Returns:
        answer: 'attract' if the force between the first and second puck is positive, 'repel' if it
         is negative and 'none' otherwise."""
    if forces[0] > 0:
        return "attract"
    elif forces[0] < 0:
        return "repel"
    return "none"


def get_configuration_answer(config):
    """Returns a concatenated string-based answer given a world configuration of masses and forces.
    Args:
        config: a length-2 list with the mass configuration in the first index and the force
        configuration in the second.
    Returns:
        answer: the concatenated answers for mass and force. See get_mass_answer and
        get_force_answer_from_flat_list."""
    masses = config[0]
    forces = config[1]

    mass_answer = get_mass_answer(masses)
    force_answer = get_force_answer_from_flat_list(forces)
    return mass_answer+"_"+force_answer


def cartesian_product(mass_all_configs, force_all_configs):
    """Finds the cartesian product between two lists.
    Args:
        mass_all_configs: all the mass configurations to be considered.
        force_all_configs: all the force configurations to be considered.
    Returns:
        all_configs: the cartesian product of the two arguments."""
    all_configs = []
    for mass_config in mass_all_configs:
        for force_config in force_all_configs:
            all_configs.append((mass_config, force_config))
    return all_configs


def generate_every_world_configuration():
    """Generates all possible world configurations as the cartesian product of the possible mass
    and force configurations.

    Returns:
        configurations: a NumPy array with all the configurations."""

    mass_all_possible = [target_pucks_mass + [1, 1] for target_pucks_mass in [[1, 1], [1, 2], [2, 1]]]
    force_all_possible = np.array(generate_possible(3, 6))
    return np.array(cartesian_product(mass_all_possible, force_all_possible))


def simulate_trial(trial):
    """Simulates a trial in the passive condition (i.e. without mouse interaction.
    Args:
        trial: the world configuration.
    Returns:
        trial_data: the simulator data after simulating the configuration.
    """

    new_env = physic_env([trial], None, None, init_mouse, T, ig_mode=0, prior=None, reward_stop=None)

    mass_answer = get_mass_answer(new_env.cond['mass'])
    force_answer = get_force_answer(new_env.cond['lf'])

    for _ in range(int(TIMEOUT/T)):
        is_done = new_env.step_passive()
        if is_done:
            break

    data = new_env.step_data()
    trial_data = pd.DataFrame()

    for object_id in ["o1", "o2", "o3", "o4"]:
        for attr in ["x", "y", "vx", "vy"]:
            trial_data[object_id+"."+attr] = data[object_id][attr]

    trial_data["A"] = mass_answer == "A"
    trial_data["B"] = mass_answer == "B"
    trial_data["same"] = mass_answer == "same"

    trial_data["attract"] = force_answer == "attract"
    trial_data["none"] = force_answer == "none"
    trial_data["repel"] = force_answer == "repel"

    return trial_data


"""def simulate_trial_in_js(trial):
    import pyduktape
    context = pyduktape.DuktapeContext()
    context.eval_js_file("simulator/js/box2d.js")
    context.eval_js_file("simulator/js/control_world.js")
    context.set_globals(cond=trial)
    context.set_globals(control_path=None)
    data = context.eval_js("Run();")

    data = json.loads(data)['physics'] #Convert to python object

    trial_data = pd.DataFrame()

    for object_id in ["o1", "o2", "o3", "o4"]:
        for attr in ["x", "y", "vx", "vy"]:
            trial_data[object_id+"."+attr] = data[object_id][attr]

    mass_answer = get_mass_answer(trial['mass'])
    force_answer = get_force_answer(trial['lf'])

    trial_data["A"] = mass_answer == "A"
    trial_data["B"] = mass_answer == "B"
    trial_data["same"] = mass_answer == "same"

    trial_data["attract"] = force_answer == "attract"
    trial_data["none"] = force_answer == "none"
    trial_data["repel"] = force_answer == "repel"

    return trial_data"""


def create_passive_datasets_for_training(n_simulations_train, n_simulations_val, n_simulations_test,
                                         train_size, val_size, test_size, train_hdf_path, val_hdf_path,
                                         test_hdf_path, n_processes, trial_hdf_key_prefix):

    """Creates .h5 datasets as the results of simulating different world configurations for the
    purposes of training a model.
    Args:
        n_simulations_train, n_simulations_val, n_simulations_test: (integer) indicates how many
        trials must be included in each dataset.

        train_size, val_size, test_size: float [0, 1), indicates the percentage of configurations
        that will be unique to the corresponding dataset. Once the configurations are divided in
        the different sets, each one can be sampled multiple times.

        train_hdf_path, val_hdf_path, test_hdf_path: (string) indicates the path where the different
        datasets must be saved.

        n_processes: (integer) indicates how many processes will be used to generate the simulations.
        That is, if n_processes > 1, the simulations for a dataset will be generated in parallel.

        trial_hdf_key_prefix: each one of the trials that are written to the .h5 file must have a
        key. This string (to which an integer will be added) will be used as prefix to the keys.
        E.g. "trial_" will generate the keys "trial_0", "trial_1", etc.
    """

    every_world_configuration = generate_every_world_configuration()
    every_world_answer = np.array(list(map(get_configuration_answer, every_world_configuration)))
    n_configurations = len(every_world_configuration)

    print(n_configurations, "possible world configurations.")
    print(str(train_size*100)+" % will be used for training "+str(val_size*100)+"% for val and "+str(test_size*100)+"% for test.")
    print("From those configurations", (n_simulations_train, n_simulations_val, n_simulations_test),
          "simulations will be produced respectively.")

    all_indices = np.arange(n_configurations)
    train_indices, not_train_indices = train_test_split(all_indices, train_size=train_size,
                                                        random_state=0, stratify=every_world_answer)
    val_indices, test_indices = train_test_split(not_train_indices, train_size=val_size / (val_size + test_size),
                                                 random_state=0,
                                                 stratify=every_world_answer[not_train_indices])

    repeated_train_indices = np.random.choice(train_indices, n_simulations_train, replace=True)
    repeated_val_indices = np.random.choice(val_indices, n_simulations_val, replace=True)
    repeated_test_indices = np.random.choice(test_indices, n_simulations_test, replace=True)

    train_cond = generate_cond(every_world_configuration[repeated_train_indices])
    val_cond = generate_cond(every_world_configuration[repeated_val_indices])
    test_cond = generate_cond(every_world_configuration[repeated_test_indices])

    for cond, path in zip([train_cond, val_cond, test_cond], [train_hdf_path, val_hdf_path, test_hdf_path]):
        if os.path.exists(path):
            os.remove(path)

        pool = Pool(processes=n_processes)
        trials = []
        for trial_i, trial in enumerate(pool.imap_unordered(simulate_trial, tqdm(cond, total=len(cond), disable=TQDM_DISABLE))):
            trials.append(trial)

        for trial_i, trial in enumerate(trials):
            trial.to_hdf(path_or_buf=path, key=trial_hdf_key_prefix+str(trial_i))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_simulations_train', type=int, default=3500)
    parser.add_argument('--n_simulations_val', type=int, default=1000)
    parser.add_argument('--n_simulations_test', type=int, default=1000)

    parser.add_argument('--train_size', type=float, default=0.7)
    parser.add_argument('--val_size', type=float, default=0.15)
    parser.add_argument('--test_size', type=float, default=0.15)

    parser.add_argument('--train_hdf_path', type=str, default="train_passive_trials.h5")
    parser.add_argument('--val_hdf_path', type=str, default="val_passive_trials.h5")
    parser.add_argument('--test_hdf_path', type=str, default="test_passive_trials.h5")

    parser.add_argument('--n_processes', type=int, default=12)

    parser.add_argument('--trial_hdf_key_prefix', type=str, default="trial_")

    args = parser.parse_args()
    create_passive_datasets_for_training(**vars(args))
