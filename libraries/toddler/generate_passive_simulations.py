from simulator.environment import physic_env
import numpy as np
from simulator.config import *
from tqdm import tqdm
import pandas as pd
import os
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

def get_mass_answer(masses):
    if masses[0] > masses[1]:
        return 'A'
    elif masses[1] > masses[0]:
        return 'B'
    return 'same'


def get_force_answer(forces):
    if forces[0][1] > 0:
        return "attract"
    elif forces[0][1] < 0:
        return "repel"
    return "none"


def get_force_answer_from_flat_list(forces):
    if forces[0] > 0:
        return "attract"
    elif forces[0] < 0:
        return "repel"
    return "none"


def simulate_trial(trial):
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


def simulate_trial_in_js(trial):
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

    return trial_data


def get_configuration_answer(config):
    masses = config[0]
    forces = config[1]

    mass_answer = get_mass_answer(masses)
    force_answer = get_force_answer_from_flat_list(forces)
    return mass_answer+"_"+force_answer


if __name__ == "__main__":
    N_SIMULATIONS_TRAIN = 3500
    N_SIMULATIONS_VAL = 1000
    N_SIMULATIONS_TEST = 1000

    every_world_configuration = generate_every_world_configuration()
    every_world_answer = np.array(list(map(get_configuration_answer, every_world_configuration)))
    n_configurations = len(every_world_configuration)

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    print(n_configurations, "possible world configurations")
    print(train_size*100, "\% will be used for training", val_size*100, "\%for val", test_size*100, "\%for test")
    print("From those configurations", (N_SIMULATIONS_TRAIN, N_SIMULATIONS_VAL, N_SIMULATIONS_TEST),
          "simulations will be produced respectively")

    all_indices = np.arange(n_configurations)
    train_indices, not_train_indices = train_test_split(all_indices, train_size=train_size,
                                                        random_state=0, stratify=every_world_answer)
    val_indices, test_indices = train_test_split(not_train_indices, train_size=0.5,
                                                 random_state=0, 
                                                 stratify=every_world_answer[not_train_indices])

    repeated_train_indices = np.random.choice(train_indices, N_SIMULATIONS_TRAIN, replace=True)
    repeated_val_indices = np.random.choice(val_indices, N_SIMULATIONS_VAL, replace=True)
    repeated_test_indices = np.random.choice(test_indices, N_SIMULATIONS_TEST, replace=True)

    train_cond = generate_cond(every_world_configuration[repeated_train_indices])
    val_cond = generate_cond(every_world_configuration[repeated_val_indices])
    test_cond = generate_cond(every_world_configuration[repeated_test_indices])

    TRAIN_PASSIVE_HDF_PATH = "train_passive_trials.h5"
    VAL_PASSIVE_HDF_PATH = "val_passive_trials.h5"
    TEST_PASSIVE_HDF_PATH = "test_passive_trials.h5"

    for cond, path in zip([train_cond, val_cond, test_cond], [TRAIN_PASSIVE_HDF_PATH, VAL_PASSIVE_HDF_PATH, TEST_PASSIVE_HDF_PATH]):
        if os.path.exists(path):
            os.remove(path)

        pool = Pool(processes=12)
        trials = [] 
        for trial_i, trial in enumerate(pool.imap_unordered(simulate_trial, tqdm(cond, total=len(cond)))):
            trials.append(trial)

        for trial_i, trial in enumerate(trials):
            trial.to_hdf(path_or_buf=path, key="trial_"+str(trial_i))
