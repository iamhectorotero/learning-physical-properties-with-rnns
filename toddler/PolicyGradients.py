import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from tqdm import tqdm

import pandas as pd
from models import ValueNetwork
from torch.autograd import Variable
import random
from copy import deepcopy
import numpy as np

from tqdm import tqdm
from simulator.environment import physic_env
from action_coding import get_mouse_action, mass_answers_idx, force_answers_idx

CHECKPOINT = 1000
MOUSE_EXPLORATION_FRAMES = 500
TIMEOUT = 1800
I_TARGET = 100 # Every 10 episodes update target network

def e_greedy_action(state, valueNetwork, epsilon, t, idx):
    possibleActions = np.arange(0, 9)

    if t < MOUSE_EXPLORATION_FRAMES:
        possibleActions = np.arange(0, 6)
    elif t == (TIMEOUT - 1):
        possibleActions = np.arange(6, 9)

    if np.random.rand() > epsilon:
        action_values = valueNetwork(state, idx=idx)[0][0][possibleActions]
        greedy_action = torch.argmax(action_values).item()
        return possibleActions[greedy_action]

    return np.random.choice(possibleActions)


def pg_action(state, policy_network, t, idx):
    probs = policy_network(state, idx=idx)
    if t == TIMEOUT - 1:
        probs[:, :, :-3] = 0.
        probs /= torch.sum(probs, dim=2)
    m = Categorical(probs)
    action = m.sample()
    policy_network.policy_log_probs.append(-m.log_prob(action))
    return int(action[0][0].numpy())

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


def store_transition(episode, state, action, reward, new_state, done):
    for i, element in enumerate([state, action, reward, new_state, done]):
            episode[i].append(element)


def train_pg(policy_network, optimizer, counter, numEpisodes, discountFactor, startingEpisode=0, mass_answers={}, force_answers={}, train_cond=(), idx=0, lock=None):

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

    agent_answers = []
    object_in_control = 0
    experience_replay = []
    SAMPLE_N_TRANSITIONS = 8
    for episodeNumber in range(startingEpisode, startingEpisode + numEpisodes):
        episode_experience = []
        epsilon = exponential_decay(counter.value)
        done = False
        frame = 0

        state = env.reset(True)
        answer = get_answer()
        answer = np.array(classes) == answer
        state = to_state_representation(state, answer=answer)

        object_A_in_control = False
        object_B_in_control = False

        episode = [[], [], [], [], []]
        while not done:
            frame += 1
            """if frame % 2 == 0:
                NO_OP = 0
                new_state, reward, done, info = env.step_active(NO_OP, get_mouse_action, question_type)
                continue"""
            action = pg_action(state, policy_network, frame, idx)
            new_state, reward, done, info = env.step_active(action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, answer=answer)

            if info["correct_answer"]:
                reward = 1.
                agent_answers.append(1)

            if info["incorrect_answer"]:
                agent_answers.append(0)
                reward = -1.

            if info["control"] == 1 and not object_A_in_control:
                reward += 0
                object_in_control += 0.5
                object_A_in_control = True

            elif info["control"] == 2 and not object_B_in_control:
                reward += 0
                object_in_control += 0.5
                object_B_in_control = True

            store_transition(episode, state, action, reward, new_state, done)
            state = new_state

        experience_replay.append(episode)
        policy_network.reset_hidden_states(idx)

        learn_pg(idx, episode[0], episode[1], episode[2], episode[3], episode[4],
                 discountFactor, policy_network)

        optimizer.step()
        optimizer.zero_grad()

        with lock:
            counter.value += 1
            if counter.value % CHECKPOINT == 0:
                print("Checkpointing Policy Network")
                saveModelNetwork(policy_network, str(counter.value)+"_model")


        upper_bound = (1 - epsilon) + epsilon / 3
        print("process %d control %d %.2f \t\taciertos %.2f (upper bound: %.2f) \teps: %.2f \tdone @ %d \tepisode %d" % 
             (idx, int(object_A_in_control) + int(object_B_in_control), object_in_control / float((episodeNumber - startingEpisode) + 1),
              np.mean(agent_answers[-100:]), upper_bound, epsilon, frame, counter.value))


def SSE(outputs, targets):
    weights = torch.ones(len(outputs[0])).cuda()
    weights[-1] = 10
    return torch.sum(weights * (outputs - targets)**2)


def learn_pg(idx, states, actions, rewards, new_states, dones, discountFactor, policy_network):
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
    returns = torch.tensor(returns)
    log_probs = torch.tensor(policy_network.policy_log_probs, requires_grad=True)

    loss = torch.sum(log_probs * returns) / len(returns)
    loss.backward()

    policy_network.policy_log_probs = []

def learn_async(idx, states, actions, rewards, new_states, dones, discountFactor, valueNetwork, target_network):
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

    # print(targets)
    """targets = rewards + discountFactor * target_network(new_states).max(dim=2)[0]
    targets = torch.where(dones, rewards, targets)
    targets = torch.where(dones, rewards, targets)
    targets = Variable(targets.clone().detach())"""

    outputs = valueNetwork(states, idx)[:, np.arange(len(states[0])), actions]
    error = nn.SmoothL1Loss()
    loss = error(targets, outputs)
    if idx == 0:
        print(loss.data)
    loss.backward()

def learn(valueNetwork, target_network, sampled_experience, discountFactor, optimizer):

    acc_loss = 0
    loss = 0
    for i, episode in enumerate(sampled_experience):
        states, actions, rewards, new_states, dones = episode
        states = torch.cat(states, dim=0)
        new_states = torch.cat(new_states, dim=0)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        # learn from full returns
        step_return = 0
        returns = []
        for i in range(len(rewards)-1, -1, -1):
            step_return = rewards[i] + discountFactor * step_return
            returns.append(step_return)
        returns.reverse()
        targets = torch.tensor(returns)

        outputs = valueNetwork(states)[:, np.arange(len(states[0])), actions]
        error = nn.SmoothL1Loss()

        loss += error(outputs, targets) / len(sampled_experience)
    print(loss.data)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


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


def validate_pg(valueNetwork, mass_answers={}, force_answers={}, val_cond=()):
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
    for episodeNumber in tqdm(range(len(val_cond) - 1)):
        done = False
        valueNetwork.reset_hidden_states(0)
        object_in_control = 0
        is_answer_correct = False

        object_A_in_control = False
        object_B_in_control = False
        frame = 0

        state = env.reset(True)
        answer = get_answer()
        answer = np.array(classes) == answer
        state = to_state_representation(state, answer=answer)

        while not done:
            frame += 1
            greedy_action = e_greedy_action(state, valueNetwork, epsilon, frame, idx=0)

            new_state, reward, done, info = env.step_active(greedy_action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, answer=answer)

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
