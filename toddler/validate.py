import os
import numpy as np
import pandas as pd
import torch

from .simulator.environment import physic_env
from .action_selection import e_greedy_action
from .RecurrentWorker import to_state_representation, remove_features_by_idx, init_mouse, store_transition
from .action_coding import get_mouse_action, CLICK, NO_OP


def validate(value_network, val_cond, timeout=1800, mass_answers={}, force_answers={}, n_bodies=4, action_repeat=1, path="replays.h5", print_stats=True, remove_features_in_index=(),
             reward_not_answering_negatively=False, reward_not_controlling_negatively=False, reward_control=False, done_with_control=False, device=torch.device("cpu"), possible_actions=np.arange(0,6), mouse_exploration_frames=None, yoked_network=None, return_replays=False, force_answer_at_t=None):
    value_network = value_network.to(device=device)

    if print_stats:
        print("----------------------------VALIDATION START-----------------------------------------")
    default_action_repeat = action_repeat

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

    epsilon = 0.00
    total_reward = 0
    total_control = []
    agent_answers = []
    lengths = []
    trials = []

    for episodeNumber in range(len(val_cond)):
        done = False
        reward = 0
        frame = 0

        state = env.reset(True, init_mouse())
        state = to_state_representation(state, frame=frame, timeout=timeout)
        state = remove_features_by_idx(state, remove_features_in_index)

        has_controlled_A = False
        has_controlled_B = False
        actions = [0]

        episode = [[], [], [], [], [], [], [], []]
        action_repeat = 0
        info = {"mouse_pos": (None, None)}
        while not done:
            frame += 1

            if action_repeat > 0:
                if greedy_action == CLICK:
                    greedy_action = NO_OP 
                action_repeat -= 1
            else:
                greedy_action = e_greedy_action(state, value_network, epsilon,
                                                frame, info["mouse_pos"],
                                                yoked_network, episode, device=device,
                                                possible_actions=possible_actions,
                                                mouse_exploration_frames=mouse_exploration_frames,
                                                force_answer_at_t=force_answer_at_t)

                action_repeat = default_action_repeat

            new_state, reward, done, info = env.step_active(greedy_action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame, timeout=timeout)
            new_state = remove_features_by_idx(new_state, remove_features_in_index)

            if info["control"] == 1 and not has_controlled_A:
                has_controlled_A = True
                if reward_control:
                    reward = 1 - float(frame) / timeout
                if done_with_control:
                    done = True

            if info["control"] == 2 and not has_controlled_B:
                has_controlled_B = True
                if reward_control:
                    reward = 1 - float(frame) / timeout

            if info["correct_answer"]:
                reward += 1
                agent_answers.append(1)
                done = True
            elif info["incorrect_answer"]:
                reward -= 1
                agent_answers.append(0)
                done = True
            elif done:
                print("NO ANSWER")
                agent_answers.append(-1)
                if reward_not_answering_negatively:
                    reward = -1
                if reward_not_controlling_negatively and not has_controlled_A and not has_controlled_B:
                    reward = -1

            if action_repeat == default_action_repeat:
                store_transition(episode, state, None, None, None, None, None, None, None)
            state = new_state
            actions.append(greedy_action)

        value_network.reset_hidden_states()

        this_episode_control = int(has_controlled_A) + int(has_controlled_B)
        total_control.append(this_episode_control)
        lengths.append(frame)

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
            print(episodeNumber+1, "Control?", this_episode_control,
                  "Correct answer?", info["correct_answer"],
                  "done @", frame)

    validation_dict = {"control": total_control, "answers": agent_answers,
                       "episode_length": lengths}

    if print_stats:
        print("Control in", np.mean(total_control), "%.", "Correct answer in", np.sum(agent_answers))
        print("----------------------------VALIDATION END-----------------------------------------")

    if return_replays:
        return validation_dict, trials
    return validation_dict
