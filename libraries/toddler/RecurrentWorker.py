import os
import time
import random
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

from copy import deepcopy
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange

from simulator.environment import physic_env
from .action_coding import get_mouse_action, mass_answers_idx, force_answers_idx
from .action_coding import CLICK, NO_OP,  MAX_X, MIN_X, MAX_Y, MIN_Y, ANSWER_QUESTION
from .action_selection import e_greedy_action
from .models import ValueNetwork

CHECKPOINT = 1000
MOUSE_EXPLORATION_FRAMES = 1
# TRAIN_YOKED_EVERY_N_EPISODES = 10


def check_for_alive_tensors():
    count = 0
    d = {}
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.size() in d:
                    d[obj.size()] += 1
                else:
                    d[obj.size()] = 1
                count += 1
        except:
            pass

    print(d)


def init_mouse():
    x = MAX_X * np.random.rand() + MIN_X
    y = MAX_Y * np.random.rand() + MIN_Y

    return (x, y)


def exponential_decay(episodeNumber, k=-0.001):
    return np.exp(k * episodeNumber)


def linear_decay_on_steps(total_steps, interval_end, min_eps=0.1):
    return max(1 - total_steps/interval_end, min_eps)


def remove_features_by_idx(state, to_remove_features=()):
    if len(to_remove_features) == 0:
        return state

    remain_features = [feature_i for feature_i in range(state.shape[-1]) if feature_i not in to_remove_features]
    state = state[:, :, remain_features]
    return state


def to_state_representation(state, frame=None, answer=None, timeout=1800):

    if answer is not None:
        state = np.hstack((state, answer.astype(np.float32)))
    elif frame is not None:
        state = np.hstack((state, np.array(frame).astype(np.float32) / timeout))
    state = state.reshape(1, 1, -1)
    return torch.from_numpy(state).float()


def store_transition(episode, state, action, reward, new_state, done, v_hh, t_hh, answer):
    for i, element in enumerate([state, action, reward, new_state, done, v_hh, t_hh, answer]):
            episode[i].append(element)


def train(value_network, optimizer, discount_factor, train_cond, n_bodies, training_data,
          total_steps, action_repeat=1, reward_control=False, done_with_control=False,
          timeout=1800, starting_episode=0, current_step=0, mass_answers={}, force_answers={}, 
          experience_replay=(), agent_answers=(), model_directory="",
          yoked_network=None, yoked_optimizer=None, device=torch.device("cpu"), remove_features_in_index=(),
          reward_not_answering_negatively=False, reward_not_controlling_negatively=False,
          possible_actions=np.arange(0,6), mouse_exploration_frames=None, force_answer_at_t=None,
          train_yoked_network_every_n_episodes=None, sample_n_episodes=32,
          experience_replay_max_size=64):

    training_data = {"question_loss":[], "value_loss":[], "control": [], "episode_length": [], "correct_answer": []}

    if len(mass_answers) > 0:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers, n_bodies=n_bodies)
        question_type = 'mass'
        get_answer = env.get_mass_true_answer
        classes = ["A is heavier", "B is heavier", "same"]
        answers_idx = mass_answers_idx
    else:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers, n_bodies=n_bodies)
        question_type = 'force'
        get_answer = env.get_force_true_answer
        classes = ["attract", "repel", "none"]
        answers_idx = force_answers_idx

    default_action_repeat = action_repeat
    total_control = []
    agent_answers = []
    experience_replay = []

    num_episodes = len(train_cond)
    pbar = tqdm(initial=starting_episode, total=starting_episode + num_episodes)
    question_loss = 0
    value_loss = 0
    accuracy = 0
    for episode_number in range(starting_episode, starting_episode + num_episodes):
        epsilon = exponential_decay(episode_number)
        done = False
        frame = 0
        has_controlled_A = False
        has_controlled_B = False

        state = env.reset(True, init_mouse())
        answer = get_answer()
        answer_idx = answers_idx[answer] - 6

        state = to_state_representation(state, frame=frame, timeout=timeout)
        state = remove_features_by_idx(state, remove_features_in_index)

        episode = [[], [], [], [], [], [], [], []]
        action_repeat = 0
        info = {"mouse_pos": (None, None)}

        while not done:
            frame += 1
            if action_repeat > 0:
                if action == CLICK:
                    action = NO_OP 
                action_repeat -= 1
            else:
                zeros = torch.zeros(value_network.n_layers, 1, value_network.hidden_dim).to(device=device)
                value_network_hh = value_network.hh if value_network.hh is not None else zeros
                action = e_greedy_action(state, value_network, epsilon, frame, info["mouse_pos"],
                                         yoked_network, episode, device=device,
                                         possible_actions=possible_actions,
                                         mouse_exploration_frames=mouse_exploration_frames,
                                         force_answer_at_t=force_answer_at_t)
                action_repeat = default_action_repeat

            reward = 0
            new_state, reward, done, info = env.step_active(action, get_mouse_action, question_type)
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
                agent_answers.append(-1)
                if reward_not_answering_negatively:
                    reward += -1
                if reward_not_controlling_negatively and not has_controlled_A and not has_controlled_B:
                    reward += -1

            if action_repeat == default_action_repeat:
                if yoked_network is not None and action >= ANSWER_QUESTION:
                    action = ANSWER_QUESTION
                store_transition(episode, state, action, reward, new_state, done, value_network_hh, None, answer_idx)
            state = new_state

        this_episode_control = int(has_controlled_A) + int(has_controlled_B)
        total_control.append(this_episode_control)
        experience_replay.append(episode)
        memory_size = len(experience_replay)
        sampled_experience_indices = np.random.choice(memory_size,
                                                      min(sample_n_episodes, memory_size),
                                                      replace=False)
        sampled_experience = [experience_replay[i] for i in sampled_experience_indices]
        value_network.reset_hidden_states()
        value_loss = learn(value_network, sampled_experience, discount_factor, optimizer, device=device)
        value_network.reset_hidden_states()

        if yoked_network is not None and episode_number != 0 and episode_number % train_yoked_network_every_n_episodes == 0:
            question_loss, accuracy = train_supervised_network(yoked_network, yoked_optimizer,
                                                               experience_replay, device=device)
            question_loss = question_loss.detach()

        training_data["value_loss"].append(value_loss)
        training_data["question_loss"].append(question_loss)
        training_data["control"].append(this_episode_control)
        training_data["episode_length"].append(frame)
        training_data["correct_answer"].append(info["correct_answer"])

        if len(experience_replay) >= experience_replay_max_size:
            experience_replay.pop(0)

        if episode_number % CHECKPOINT == 0:
            # print("Checkpointing ValueNetwork")
            saveModelNetwork(value_network, model_directory+str(episode_number)+"_model")
            if yoked_network is not None:
                saveModelNetwork(yoked_network, model_directory+str(episode_number)+"_yokednet")

        upper_bound = (1 - epsilon) + epsilon / 3
        desc = "control %.2f aciertos %.2f (UB: %.2f) eps: %.2f done @ %d vloss %.6f qloss %.6f accuracy %.2f" % (np.mean(total_control[-500:]), np.mean(agent_answers[-500:]), upper_bound, epsilon, frame, value_loss, question_loss, accuracy)
        pbar.update(1)
        pbar.set_description(desc)

        current_step += frame
        if current_step >= total_steps:
            break

    saveModelNetwork(value_network, model_directory+str(episode_number)+"_model")
    if yoked_network is not None:
        saveModelNetwork(yoked_network, model_directory+str(episode_number)+"_yokednet")

    pbar.close()
    return training_data


def learn(valueNetwork, sampled_experience, discountFactor, optimizer,
          seq_length=600, learn_from_sequence_end=True, device=torch.device("cpu")):

    valueNetwork = valueNetwork.train()
    batch_states = []
    batch_actions = []
    batch_returns = []
    batch_new_states = []
    batch_dones = []
    batch_value_hidden_states = []
    batch_answers = []

    episode_length = [len(states) for states, _, _, _, _, _, _, _ in sampled_experience]
    min_episode_length = min(episode_length)
    seq_length = min(seq_length, min_episode_length)

    for i, episode in enumerate(sampled_experience):
        states, actions, rewards, new_states, dones, value_hh, target_hh, answer = episode

        if len(states) == seq_length:
            starting_idx = 0
        else:
            if learn_from_sequence_end:
                starting_idx = len(states) - seq_length
            else:
                starting_idx = np.random.randint(0, len(states)-seq_length)
        end_idx = starting_idx + seq_length

        batch_states.append(torch.cat(states, dim=1)[:, starting_idx:end_idx, :])
        batch_actions.append(actions[starting_idx:end_idx])
        batch_new_states.append(torch.cat(new_states, dim=1)[:, starting_idx:end_idx, :])
        batch_dones.append(dones[starting_idx:end_idx])
        batch_value_hidden_states.append(value_hh[starting_idx])
        batch_answers.append(answer[0])

        step_return = 0
        returns = []
        for i in range(len(rewards)-1, -1, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        batch_returns.append(returns[starting_idx:end_idx])

    if len(batch_actions) == 0:
        return 0.

    states = torch.cat(batch_states, dim=0).to(device=device)
    new_states = torch.cat(batch_new_states, dim=0).to(device=device)
    value_hidden_states = torch.cat(batch_value_hidden_states, dim=1).to(device=device)
    targets = torch.tensor(batch_returns).to(device=device)
    dones = torch.tensor(batch_dones).to(device=device)
    actions = batch_actions

    valueNetwork.reset_hidden_states(value_hidden_states.detach())
    every_action_value = valueNetwork.predict(states, value_hidden_states)

    outputs = [every_action_value[i, np.arange(every_action_value.shape[1]), actions[i]] for i in range(every_action_value.shape[0])]
    outputs = torch.stack(outputs)
    error = nn.MSELoss().to(device=device)
    loss = error(outputs, targets)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.detach()


def learn_example_by_example(valueNetwork, sampled_experience, discountFactor, optimizer,
                             seq_length=600, learn_from_sequence_end=True, device=torch.device("cpu")):

    acc_loss = 0

    for i, episode in enumerate(sampled_experience):
        states, actions, rewards, new_states, dones, value_hh, target_hh = episode

        if len(states) <= seq_length:
            starting_idx = 0
        else:
            if learn_from_sequence_end:
                starting_idx = len(states) - seq_length
            else:
                starting_idx = np.random.randint(0, len(states)-seq_length)
        end_idx = starting_idx + seq_length

        states = torch.cat(states, dim=1)[:, starting_idx:end_idx, :].to(device=device)
        actions = torch.tensor(actions[starting_idx:end_idx]).to(device=device)
        new_states = torch.cat(new_states, dim=1)[:, starting_idx:end_idx, :].to(device=device)
        dones = torch.tensor(dones[starting_idx:end_idx]).to(device=device)
        hidden_state = value_hh[starting_idx].to(device=device)

        step_return = 0
        returns = []
        for i in range(len(rewards)-1, -1, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        returns = torch.tensor(returns[starting_idx:end_idx]).to(device=device)

        valueNetwork.reset_hidden_states(hidden_state.detach())
        every_action_value = valueNetwork(states)
        outputs = every_action_value[0, np.arange(every_action_value.shape[1]), actions]
        error = nn.MSELoss().to(device=device)
        loss = error(outputs, returns) / len(sampled_experience) * len(states)
        loss.backward()
        acc_loss += loss
    optimizer.step()
    optimizer.zero_grad()

    return acc_loss.data


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def train_supervised_network(yoked_network, yoked_optimizer, experience_replay, device=torch.device("cpu")):
    error = nn.CrossEntropyLoss().to(device=device)
    episodes = [episode[0] for episode in experience_replay]
    targets = [episode[-1][0] for episode in experience_replay]

    yoked_network = yoked_network.train()
    training_idx = np.arange(len(experience_replay)//2)
    validation_idx = np.arange(len(experience_replay)//2, len(experience_replay))

    features = [0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15]
    X_train = torch.cat([torch.cat(episodes[i][:300], dim=1) for i in training_idx], dim=0).to(device=device)
    Y_train = torch.tensor([targets[i] for i in training_idx]).to(device=device)

    X_val = torch.cat([torch.cat(episodes[i][:300], dim=1) for i in validation_idx], dim=0).to(device)
    Y_val = torch.tensor([targets[i] for i in validation_idx]).to(device=device)

    X_train = X_train[:, :, features]
    X_val = X_val[:, :, features]

    total_loss = 0
    EPOCHS = 10
    for epoch in range(EPOCHS):
        epoch_loss = 0.
        y_hat = yoked_network(X_train)
        loss = error(y_hat, Y_train)
        loss.backward()
        yoked_optimizer.step()
        yoked_optimizer.zero_grad()
        total_loss += loss.data
    total_loss /= EPOCHS

    yoked_network = yoked_network.eval()

    y_hat = torch.argmax(yoked_network(X_val), dim=1)
    correct_answers = (y_hat == Y_val).sum()
    accuracy = float(correct_answers) / len(Y_val)

    return total_loss, accuracy
