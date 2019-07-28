import os
import numpy as np
import pandas as pd

from simulator.environment import physic_env
from .RecurrentWorker import no_answers_e_greedy_action, to_state_representation, remove_features_by_idx, init_mouse
from .action_coding import get_mouse_action, CLICK, NO_OP


def validate(valueNetwork, mass_answers={}, force_answers={}, val_cond=(), n_bodies=4, action_repeat=1, path="replays.h5", print_stats=True):
    if print_stats:
        print("----------------------------VALIDATION START-----------------------------------------")
    action_repeat_default = action_repeat

    for cond in val_cond:
        cond['timeout'] = 1800

    if len(mass_answers) > 0:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers, n_bodies=n_bodies)
        question_type = 'mass'
        get_answer = env.get_mass_true_answer
        classes = ["A is heavier", "B is heavier", "same"]
    else:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers, n_bodies=n_bodies)
        question_type = 'force'
        get_answer = env.get_force_true_answer
        classes = ["attract", "repel", "none"]

    if os.path.exists("replays.h5"):
        os.remove("replays.h5")

    epsilon = 0.00
    total_reward = 0
    trials = []

    for episodeNumber in range(len(val_cond)):
        done = False
        reward = 0
        frame = 0

        state = env.reset(True, init_mouse())
        state = to_state_representation(state, frame=frame)
        state = remove_features_by_idx(state, [2, 3])

        actions = [0]
        action_repeat = 0
        info = {"mouse_pos": (None, None)}
        while not done:
            frame += 1

            if action_repeat > 0:
                if greedy_action == CLICK:
                    greedy_action = NO_OP 
                action_repeat -= 1
            else:
                greedy_action = no_answers_e_greedy_action(state, valueNetwork, epsilon, frame, info["mouse_pos"])
                action_repeat = action_repeat_default

            new_state, reward, done, info = env.step_active(greedy_action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame)
            new_state = remove_features_by_idx(new_state, [2, 3])

            if info["control"]  == 1:
                reward = 1
                total_reward += 1
                done = True

            state = new_state
            actions.append(greedy_action)

        valueNetwork.reset_hidden_states()

        data = env.step_data()
        trial_data = pd.DataFrame()

        for object_id in ["o1", "o2", "o3", "o4"][:n_bodies]:
            for attr in ["x", "y", "vx", "vy"]:
                trial_data[object_id+"."+attr] = data[object_id][attr]

        trial_data["actions"] = actions
        trial_data["mouseX"] = data["mouse"]["x"]
        trial_data["mouseY"] = data["mouse"]["y"]
        trial_data["mouse.vx"] = data["mouse"]["vx"]
        trial_data["mouse.vy"] = data["mouse"]["vy"]
        trial_data["idControlledObject"] = ["none" if obj == 0 else "object"+str(obj) for obj in data["co"]]
        trials.append(trial_data)

        if print_stats:
            print(episodeNumber+1, "Control?", reward == 1, "done @", frame)
    accuracy = total_reward / len(val_cond)
    if print_stats:
        print("Success in", accuracy*100, "%")
        print("----------------------------VALIDATION END-----------------------------------------")

    if path is not None:
        for i, trial_data in enumerate(trials):
            trial_data.to_hdf(path, key="episode_"+str(i))

    return accuracy, trials
