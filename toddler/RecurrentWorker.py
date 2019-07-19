import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from tqdm import tqdm
import ipdb

import pandas as pd
from models import ValueNetwork
from torch.autograd import Variable
import random
from copy import deepcopy
import numpy as np

from tqdm import tqdm, trange
from simulator.environment import physic_env
from action_coding import get_mouse_action, mass_answers_idx, force_answers_idx

CHECKPOINT = 1000
MOUSE_EXPLORATION_FRAMES = 600
TIMEOUT = 1800
I_TARGET = 35 # Every 10 episodes update target network

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


def exponential_decay(episodeNumber, k=-0.001):
    return np.exp(k * episodeNumber)


def to_state_representation(state, frame=None, answer=None):

    if answer is not None:
        # state = answer.astype(np.float32)
        state = np.hstack((state, answer.astype(np.float32)))
    elif frame is not None:
        state = np.array(state.tolist() + [float(frame)])
    state = state.reshape(1, 1, -1)
    return torch.from_numpy(state).float()


def store_transition(episode, state, action, reward, new_state, done, v_hh, t_hh):
    for i, element in enumerate([state, action, reward, new_state, done, v_hh, t_hh]):
            episode[i].append(element)


def train(valueNetwork, target_network, optimizer, counter, numEpisodes, discountFactor, startingEpisode=0, mass_answers={}, force_answers={}, train_cond=(), idx=0, lock=None, experience_replay=(), agent_answers=()):

    np.random.seed(idx)
    for cond in train_cond:
        cond['timeout'] = TIMEOUT

    if len(mass_answers) > 0:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers)
        question_type = 'mass'
        get_answer = env.get_mass_true_answer
        classes = ["A is heavier", "B is heavier", "same"]
    else:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers)
        question_type = 'force'
        get_answer = env.get_force_true_answer
        classes = ["attract", "repel", "none"]

    agent_answers = list(agent_answers)
    experience_replay = list(experience_replay)
    epsilon = exponential_decay(counter.value)
    object_in_control = 0
    SAMPLE_N_EPISODES = 64

    pbar = tqdm(initial=startingEpisode, total=startingEpisode + numEpisodes)
    for episodeNumber in range(startingEpisode, startingEpisode + numEpisodes):
        epsilon = exponential_decay(counter.value)
        done = False
        frame = 0

        state = env.reset(True)
        answer = get_answer()
        answer = np.array(classes) == answer
        state = to_state_representation(state, frame=frame)

        object_A_in_control = False
        object_B_in_control = False

        episode = [[], [], [], [], [], [], []]
        loss = 0
        while not done:
            frame += 1
            if frame % 2 == 0:
                NO_OP = 0
                new_state, reward, done, info = env.step_active(NO_OP, get_mouse_action, question_type)
                new_state = to_state_representation(new_state, answer=answer)
                continue

            value_network_hh = valueNetwork.hh if valueNetwork.hh is not None else torch.zeros(1, 1, 1000).cuda()
            action = e_greedy_action(state, valueNetwork, epsilon, frame, None)
            new_state, reward, done, info = env.step_active(action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame)

            if info["correct_answer"]:
                reward = 1.
                agent_answers.append(1)

            if info["incorrect_answer"]:
                agent_answers.append(0)
                reward = -1

            if info["control"] == 1 and not object_A_in_control:
                reward += 0
                object_in_control += 0.5
                object_A_in_control = True

            elif info["control"] == 2 and not object_B_in_control:
                reward += 0
                object_in_control += 0.5
                object_B_in_control = True

            store_transition(episode, state, action, reward, new_state, done, value_network_hh, None)
            state = new_state

        experience_replay.append(episode)
        if len(experience_replay) >= SAMPLE_N_EPISODES:
            memory_size = len(experience_replay)
            sampled_experience_indices = np.random.choice(memory_size,
                                                          SAMPLE_N_EPISODES,
                                                          replace=False)
            sampled_experience = [experience_replay[i] for i in sampled_experience_indices]
            # sampled_experience = experience_replay[-SAMPLE_N_EPISODES:]
            valueNetwork.reset_hidden_states()
            loss = learn(valueNetwork, target_network, sampled_experience, discountFactor, optimizer)

            if len(experience_replay) == SAMPLE_N_EPISODES:
                experience_replay.pop(0)

        valueNetwork.reset_hidden_states()

        with lock:
            counter.value += 1
            """if counter.value % I_TARGET == 0:
                print("Updating Target Network")
                target_network.load_state_dict(valueNetwork.state_dict())"""
            if counter.value % CHECKPOINT == 0:
                print("Checkpointing ValueNetwork")
                saveModelNetwork(valueNetwork, str(counter.value)+"_model")

        upper_bound = (1 - epsilon) + epsilon / 3
        desc = "control %.2f aciertos %.2f (UB: %.2f) eps: %.2f done @ %d loss %.6f" % (object_in_control / float((episodeNumber - startingEpisode) + 1), np.mean(agent_answers[-100:]), upper_bound, epsilon, frame, loss)
        pbar.update(1)
        pbar.set_description(desc)

    pbar.close()
    return experience_replay, agent_answers

def SSE(outputs, targets):
    weights = torch.ones(len(outputs[0])).cuda()
    weights[-1] = 10
    return torch.sum(weights * (outputs - targets)**2)


"""def learn_async(idx, states, actions, rewards, new_states, dones, discountFactor, valueNetwork, target_network):
    new_states = torch.cat(new_states, dim=1)
    states = torch.cat(states, dim=1)
    dones = torch.tensor(dones)
    # learn from full returns
    step_return = 0
    returns = []
    for i in range(len(rewards)-1, -1, -1):
        step_return = rewards[i] + discountFactor * step_return
        returns.append(step_return)
    returns.reverse()
    targets = torch.tensor(returns)

    outputs = valueNetwork(states, idx)[:, np.arange(len(states[0])), actions]
    error = nn.MSELoss()
    loss = error(targets, outputs)
    if idx == 0:
        print(loss.data)
    loss.backward()
"""

def learn(valueNetwork, target_network, sampled_experience, discountFactor, optimizer,
          seq_length=900):

    acc_loss = 0
    loss = 0
    batch_states = []
    batch_actions = []
    batch_returns = []
    batch_new_states = []
    batch_dones = []
    batch_value_hidden_states = []
    # batch_target_hidden_states = []

    episode_length = [len(states) for states, _, _, _, _, _, _ in sampled_experience]
    min_episode_length = min(episode_length)
    # print(episode_length, min_episode_length)
    seq_length = min(seq_length, min_episode_length)
    starting_idx = -seq_length
    end_idx = None

    for i, episode in enumerate(sampled_experience):
        states, actions, rewards, new_states, dones, value_hh, target_hh = episode

        batch_states.append(torch.cat(states, dim=1)[:, starting_idx:end_idx, :])
        batch_actions.append(actions[starting_idx:end_idx])
        batch_new_states.append(torch.cat(new_states, dim=1)[:, starting_idx:end_idx, :])
        batch_dones.append(dones[starting_idx:end_idx])
        batch_value_hidden_states.append(value_hh[starting_idx])
        # batch_target_hidden_states.append(target_hh[starting_idx])

        step_return = 0
        returns = []
        for i in range(len(rewards)-1, len(rewards)-1-seq_length, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        batch_returns.append(returns[starting_idx:end_idx])

    """print(batch_states[0].shape)
    print(batch_actions[0].shape)
    print(batch_rewards[0].shape)
    print(batch_new_states[0].shape)
    print(batch_dones[0].shape)
    print(batch_hidden_states[0].shape)"""

    actions = torch.tensor(batch_actions)
    states = torch.cat(batch_states, dim=0).cuda()
    new_states = torch.cat(batch_new_states, dim=0).cuda()
    targets = torch.tensor(batch_returns).cuda()
    dones = torch.tensor(batch_dones).cuda()
    value_hidden_states = torch.cat(batch_value_hidden_states, dim=1).cuda()
    # target_hidden_states = torch.cat(batch_target_hidden_states, dim=1).cuda()

    # target_network.reset_hidden_states(target_hidden_states)
    valueNetwork.reset_hidden_states(value_hidden_states.detach())

    # next_state_values = target_network(new_states).max(dim=2)[0]
    # targets = rewards + discountFactor * next_state_values
    # targets = torch.where(dones, rewards, targets)
    # targets = Variable(targets.clone().detach())

    every_action_value = valueNetwork(states)
    outputs = [every_action_value[i, np.arange(every_action_value.shape[1]), actions[i]] for i in range(every_action_value.shape[0])]
    outputs = torch.stack(outputs)
    error = nn.MSELoss().cuda()

    """if states.shape[0] > 1:
        print(every_action_value)
        print(actions)
        print(outputs)
        print(outputs.shape)
        print(targets.shape)
        # ipdb.set_trace()
        # exit()"""

    loss = error(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss.data

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
    if done:
        return torch.tensor(reward)

    target = torch.tensor(reward + discountFactor * targetNetwork(nextObservation)[0].max())

    return target


def computePrediction(state, action, valueNetwork):
    return valueNetwork(state)[0][action]


# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
    torch.save(model.state_dict(), strDirectory)


def validate(valueNetwork, mass_answers={}, force_answers={}, val_cond=()):
    print("----------------------------VALIDATION START-----------------------------------------")


    if len(mass_answers) > 0:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers)
        question_type = 'mass'
        get_answer = env.get_mass_true_answer
        classes = ["A is heavier", "B is heavier", "same"]
    else:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers)
        question_type = 'force'
        get_answer = env.get_force_true_answer
        classes = ["attract", "repel", "none"]

    if os.path.exists("replays.h5"):
        os.remove("replays.h5")

    epsilon = 0.01
    correct_answers = 0

    answers = []
    ground_truth = []
    for episodeNumber in range(len(val_cond) - 1):
        done = False
        object_in_control = 0
        is_answer_correct = False

        object_A_in_control = False
        object_B_in_control = False
        frame = 0

        state = env.reset(True)
        answer = get_answer()
        answer = np.array(classes) == answer
        state = to_state_representation(state, frame=frame)

        while not done:
            frame += 1
            greedy_action = e_greedy_action(state, valueNetwork, epsilon, frame)

            new_state, reward, done, info = env.step_active(greedy_action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame)

            if info["correct_answer"]:
                is_answer_correct = True
                correct_answers += 1
            if info["correct_answer"] or info["incorrect_answer"]:
                print(classes[greedy_action % 6], get_answer())
            if info["control"]  == 1 and not object_A_in_control:
                reward += 1
                object_in_control += 0.5
                object_A_in_control = True
            if info["control"]  == 2 and not object_B_in_control:
                reward += 1
                object_in_control += 0.5
                object_B_in_control = True

            state = new_state
        valueNetwork.reset_hidden_states()

        data = env.step_data()
        trial_data = pd.DataFrame()

        for object_id in ["o1", "o2", "o3", "o4"]:
            for attr in ["x", "y", "vx", "vy"]:
                trial_data[object_id+"."+attr] = data[object_id][attr]

        trial_data["ground_truth"] = get_answer()
        trial_data["answer"] = classes[greedy_action % 6]
        trial_data["mouseX"] = data["mouse"]["x"]
        trial_data["mouseY"] = data["mouse"]["y"]
        trial_data["mouse.vx"] = data["mouse"]["vx"]
        trial_data["mouse.vy"] = data["mouse"]["vy"]
        trial_data["idControlledObject"] = ["none" if obj == 0 else "object"+str(obj) for obj in data["co"]]
        trial_data.to_hdf("replays.h5", key="episode_"+str(episodeNumber))

        print(episodeNumber, "Correct?", is_answer_correct, "Control", object_in_control, "done @", frame)
    print("Correct perc", correct_answers / 10)
    print("----------------------------VALIDATION END-----------------------------------------")
