import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import ipdb

import pandas as pd
from .models import ValueNetwork
from torch.autograd import Variable
import random
from copy import deepcopy
import numpy as np

from tqdm import tqdm, trange
from simulator.environment import physic_env
from .action_coding import get_mouse_action, mass_answers_idx, force_answers_idx
from .action_coding import MIN_X, MAX_X, MIN_Y, MAX_Y
from .action_coding import CLICK, NO_OP, ACCELERATE_IN_X, ACCELERATE_IN_Y, DECELERATE_IN_X, DECELERATE_IN_Y

ACTION_REPEAT = 1
CHECKPOINT = 1000
MOUSE_EXPLORATION_FRAMES = 2
TIMEOUT = 600

def init_mouse():
    x = MAX_X * np.random.rand() + MIN_X
    y = MAX_Y * np.random.rand() + MIN_Y

    return (x, y)

def e_greedy_action(state, valueNetwork, epsilon, t, target_network=None):
    if target_network is not None:
        target_network(state.cuda())

    possibleActions = np.arange(0, 9)
    action_values = valueNetwork(state.cuda())[0][0]
    greedy_action = torch.argmax(action_values).item()

    policy = []
    for i, a in enumerate(possibleActions):
        if i == greedy_action:
            policy.append(1 - epsilon + epsilon/len(possibleActions))
        else:
            policy.append(epsilon/len(possibleActions))

    policy = np.array(policy)
    if t < MOUSE_EXPLORATION_FRAMES:
        policy[-3:] = 0
        policy /= np.sum(policy)
    elif t == (TIMEOUT - 1):
        policy[:-3] = 0
        policy /= np.sum(policy)

    return np.random.choice(possibleActions, p=policy)


def no_answers_e_greedy_action(state, valueNetwork, epsilon, t, current_pos=(None, None)):
    possibleActions = np.arange(0, 6)
    action_values = valueNetwork(state.cuda())[0][0]
    greedy_action = torch.argmax(action_values).item()

    policy = []
    for i, a in enumerate(possibleActions):
        if i == greedy_action:
            policy.append(1 - epsilon + epsilon/len(possibleActions))
        else:
            policy.append(epsilon/len(possibleActions))

    x_pos, y_pos = current_pos
    if x_pos == MAX_X:
        policy[ACCELERATE_IN_X] = 0
    elif x_pos == MIN_X:
        policy[DECELERATE_IN_X] = 0

    if y_pos == MAX_Y:
        policy[ACCELERATE_IN_Y] = 0
    elif y_pos == MIN_Y:
        policy[DECELERATE_IN_Y] = 0

    policy = np.array(policy) / sum(policy)

    return np.random.choice(possibleActions, p=policy)


def exponential_decay(episodeNumber, k=-0.0001):
    return np.exp(k * episodeNumber)


def remove_features_by_idx(state, to_remove_features=()):
        remain_features = [feature_i for feature_i in range(state.shape[-1]) if feature_i not in to_remove_features]
        state = state[:, :, remain_features]
        return state


def to_state_representation(state, frame=None, answer=None, timeout=TIMEOUT):

    if answer is not None:
        state = np.hstack((state, answer.astype(np.float32)))
    elif frame is not None:
        state = np.hstack((state, np.array(frame).astype(np.float32) / timeout))
    state = state.reshape(1, 1, -1)
    return torch.from_numpy(state).float()


def store_transition(episode, state, action, reward, new_state, done, v_hh, t_hh):
    for i, element in enumerate([state, action, reward, new_state, done, v_hh, t_hh]):
            episode[i].append(element)


def train(valueNetwork, optimizer, numEpisodes, discountFactor, startingEpisode=0, mass_answers={}, force_answers={}, train_cond=(), experience_replay=(), agent_answers=(), n_bodies=4, training_data={}, starting_puck_speed=10.):

    np.random.seed(42)
    for cond in train_cond:
        cond['timeout'] = TIMEOUT
        # vs = np.random.uniform(-starting_puck_speed, starting_puck_speed, (n_bodies, 2))
        # cond['svs'] = [{"x": vs[i][0], "y": vs[i][1]} for i in range(n_bodies)]

    if len(mass_answers) > 0:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers, n_bodies=n_bodies)
        question_type = 'mass'
        get_answer = env.get_mass_true_answer
        classes = ["A is heavier", "B is heavier", "same"]
    else:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers, n_bodies=n_bodies)
        question_type = 'force'
        get_answer = env.get_force_true_answer
        classes = ["attract", "repel", "none"]

    agent_answers = list(agent_answers)
    experience_replay = list(experience_replay)
    total_reward = 0
    SAMPLE_N_EPISODES = 32

    pbar = tqdm(initial=startingEpisode, total=startingEpisode + numEpisodes)
    for episodeNumber in range(startingEpisode, startingEpisode + numEpisodes):
        epsilon = exponential_decay(episodeNumber)
        done = False
        frame = 0

        state = env.reset(True, init_mouse())
        answer = get_answer()
        answer = np.array(classes) == answer
        state = to_state_representation(state, frame=frame)
        state = remove_features_by_idx(state, [2, 3])

        episode = [[], [], [], [], [], [], []]
        loss = 0
        action_repeat = 0
        info = {"mouse_pos": (None, None)}

        while not done:
            frame += 1
            if action_repeat > 0:
                if action == CLICK:
                    action = NO_OP 
                action_repeat -= 1
            else:
                zeros = torch.zeros(4, 1, 40).cuda()
                value_network_hh = valueNetwork.hh if valueNetwork.hh is not None else zeros
                action = no_answers_e_greedy_action(state, valueNetwork, epsilon, frame,
                                                    info["mouse_pos"])
                action_repeat = ACTION_REPEAT

            new_state, reward, done, info = env.step_active(action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame)
            new_state = remove_features_by_idx(new_state, [2, 3])

            if info["control"] == 1:
                reward = 1 - float(frame) / TIMEOUT
                # print(reward)
                total_reward += 1.
                done = True
            elif done:
                reward = -1

            if action_repeat == ACTION_REPEAT:
                store_transition(episode, state, action, reward, new_state, done, value_network_hh, None)
            state = new_state

        experience_replay.append(episode)
        memory_size = len(experience_replay)
        sampled_experience_indices = np.random.choice(memory_size,
                                                      min(SAMPLE_N_EPISODES, memory_size),
                                                      replace=False)
        sampled_experience = [experience_replay[i] for i in sampled_experience_indices]
        valueNetwork.reset_hidden_states()
        loss = learn(valueNetwork, sampled_experience, discountFactor, optimizer)

        training_data["loss"].append(float(loss.cpu().numpy()))
        training_data["control"].append(info["control"])
        training_data["episode_length"].append(frame)

        if len(experience_replay) >= 2 * SAMPLE_N_EPISODES:
            experience_replay.pop(0)

        valueNetwork.reset_hidden_states()

        if episodeNumber % CHECKPOINT == 0:
            print("Checkpointing ValueNetwork")
            saveModelNetwork(valueNetwork, str(episodeNumber)+"_model")

        agent_answers = [0]
        upper_bound = (1 - epsilon) + epsilon / 3
        desc = "control %.2f aciertos %.2f (UB: %.2f) eps: %.2f done @ %d loss %.6f" % (total_reward/ float((episodeNumber - startingEpisode) + 1), np.mean(agent_answers), upper_bound, epsilon, frame, loss)
        pbar.update(1)
        pbar.set_description(desc)

    pbar.close()
    return experience_replay, agent_answers, training_data


def learn(valueNetwork, sampled_experience, discountFactor, optimizer,
          seq_length=600, learn_from_sequence_end=True):

    acc_loss = 0
    loss = 0
    batch_states = []
    batch_actions = []
    batch_returns = []
    batch_new_states = []
    batch_dones = []
    batch_value_hidden_states = []

    episode_length = [len(states) for states, _, _, _, _, _, _ in sampled_experience]
    min_episode_length = min(episode_length)
    seq_length = min(seq_length, min_episode_length)

    for i, episode in enumerate(sampled_experience):
        states, actions, rewards, new_states, dones, value_hh, target_hh = episode

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
        # batch_target_hidden_states.append(target_hh[starting_idx])

        step_return = 0
        returns = []
        for i in range(len(rewards)-1, -1, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        batch_returns.append(returns[starting_idx:end_idx])

    if len(batch_actions) == 0:
        return 0.

    actions = torch.tensor(batch_actions)
    states = torch.cat(batch_states, dim=0).cuda()
    new_states = torch.cat(batch_new_states, dim=0).cuda()
    targets = torch.tensor(batch_returns).cuda()
    dones = torch.tensor(batch_dones).cuda()
    value_hidden_states = torch.cat(batch_value_hidden_states, dim=1).cuda()

    valueNetwork.reset_hidden_states(value_hidden_states.detach())

    every_action_value = valueNetwork(states)
    outputs = [every_action_value[i, np.arange(every_action_value.shape[1]), actions[i]] for i in range(every_action_value.shape[0])]
    outputs = torch.stack(outputs)
    error = nn.MSELoss().cuda()

    loss = error(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data


def learn_example_by_example(valueNetwork, sampled_experience, discountFactor, optimizer,
                             seq_length=80, learn_from_sequence_end=True):

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

        states = torch.cat(states, dim=1)[:, starting_idx:end_idx, :].cuda()
        actions = torch.tensor(actions[starting_idx:end_idx]).cuda()
        new_states = torch.cat(new_states, dim=1)[:, starting_idx:end_idx, :].cuda()
        dones = torch.tensor(dones[starting_idx:end_idx]).cuda()
        hidden_state = value_hh[starting_idx].cuda()

        step_return = 0
        returns = []
        for i in range(len(rewards)-1, -1, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        returns = torch.tensor(returns[starting_idx:end_idx]).cuda()

        valueNetwork.reset_hidden_states(hidden_state.detach())
        every_action_value = valueNetwork(states)
        outputs = every_action_value[0, np.arange(every_action_value.shape[1]), actions]
        error = nn.MSELoss().cuda()
        loss = error(outputs, returns) / len(sampled_experience)
        loss.backward()
        acc_loss += loss
    optimizer.step()
    optimizer.zero_grad()

    return acc_loss.data


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


