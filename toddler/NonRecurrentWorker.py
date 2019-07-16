import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

import pandas as pd
from models import ValueNetwork
from torch.autograd import Variable
import random
from copy import deepcopy
import numpy as np

from tqdm import tqdm
from simulator.environment import physic_env
from action_coding import get_mouse_action, mass_answers_idx, force_answers_idx
from isaac.constants import YOKED_TRAINING_COLS, FORCE_CLASS_COLS, MASS_CLASS_COLS, MOUSE_CONTROL_COLS
from isaac.models import initialise_model
from isaac.training import training_loop, evaluate

CHECKPOINT = 100000
MOUSE_EXPLORATION_FRAMES = 1000
TIMEOUT = 1800
I_TARGET = MOUSE_EXPLORATION_FRAMES * 100 # Every 100 episodes update target network

def e_greedy_action(state, valueNetwork, epsilon, t):
    possibleActions = np.arange(0, 6)

    if np.random.rand() > epsilon:
        action_values = valueNetwork(state, online=True)[0][0][possibleActions]
        greedy_action = torch.argmax(action_values).item()
        return possibleActions[greedy_action]

    return np.random.choice(possibleActions)


def exponential_decay(episodeNumber, k=-0.001):
    return np.exp(k * episodeNumber)


def to_state_representation(state, frame=None, answer=None):

    if answer is not None:
        state = np.hstack((state, answer.astype(np.float32)))
    elif frame is not None:
        state = np.array(state.tolist() + [float(frame)])
    state = state.reshape(1, -1)
    return torch.from_numpy(state).float()


def train(valueNetwork, target_network, optimizer, counter, numEpisodes, discountFactor, startingEpisode=0, mass_answers={}, force_answers={}, train_cond=()):

    TR_COLS = YOKED_TRAINING_COLS + MOUSE_CONTROL_COLS
    network_dims = (len(TR_COLS), 25, 4, 3)
    supervised_network, error, optimizer = initialise_model(network_dims, dropout=0.5, lr=0.01)

    for cond in train_cond:
        cond['timeout'] = TIMEOUT

    if len(mass_answers) > 0:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers)
        question_type = 'mass'
        get_answer_method = env.get_mass_true_answer
    else:
        env = physic_env(train_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers)
        question_type = 'force'
        get_answer_method = env.get_force_true_answer

    experience_replay = []
    SAMPLE_N_TRANSITIONS = 8

    agent_answers = []
    object_in_control = 0

    for episodeNumber in range(startingEpisode, startingEpisode + numEpisodes):
        target_network = target_network.cpu()
        valueNetwork = valueNetwork.cpu()

        episode_experience = []
        epsilon = exponential_decay(episodeNumber)
        done = False
        frame = 0

        state = env.reset(True)
        answer = env.get_force_true_answer()
        answer = np.array(["attract", "repel", "none"]) == answer
        state = to_state_representation(state, frame=frame)

        object_A_in_control = False
        object_B_in_control = False

        while not done:
            frame += 1

            action = e_greedy_action(state, valueNetwork, epsilon, frame)
            new_state, reward, done, info = env.step_active(action, get_mouse_action, question_type)
            new_state = to_state_representation(new_state, frame=frame)

            if done:
                supervised_data.append((env.step_data(), get_answer_method()))
                reward += get_prediction_reward(supervised_network, supervised_data[-1])
            if info["control"] == 1 and not object_A_in_control:
                reward += 0
                object_in_control += 0.5
                object_A_in_control = True

            elif info["control"] == 2 and not object_B_in_control:
                reward += 0
                object_in_control += 0.5
                object_B_in_control = True

            episode_experience.append((state[0], action, reward, new_state[0], done))
            experience_replay.append((state[0], action, reward, new_state[0], done))
            state = new_state

            counter += 1
            if counter % I_TARGET == 0:
                print("Updating Target Network")
                target_network.load_state_dict(valueNetwork.state_dict())
            if counter % CHECKPOINT == 0:
                print("Checkpointing ValueNetwork")
                saveModelNetwork(valueNetwork, str(counter)+"_model")

            if counter > 32e6:
                exit()

            sampled_experience_indices = np.random.choice(len(experience_replay),
                                                          min(len(experience_replay), SAMPLE_N_TRANSITIONS) ,
                                                          replace=False)
            sampled_experience = [experience_replay[i] for i in sampled_experience_indices]
            # sampled_experience = np.random.shuffle(np.concatenate(sampled_experience))
            learn(valueNetwork, target_network, sampled_experience, discountFactor, optimizer)

        upper_bound = (1 - epsilon) + epsilon / 3
        print("control %d %.2f \t\taciertos %.2f (upper bound: %.2f) \teps: %.2f \tdone @ %d \tepisode %d" % 
             (int(object_A_in_control) + int(object_B_in_control), object_in_control / float((episodeNumber - startingEpisode) + 1),
              np.mean(agent_answers[-100:]), upper_bound, epsilon, frame, episodeNumber))

        experience_replay.append(episode_experience)
        if len(experience_replay) > 256:
            experience_replay.pop(0)

        if episodeNumber != 0 and episodeNumber % 10 == 0:
            print("supervised learning")
            supervised_learning(supervised_network, error, optimizer, supervised_data,
                                question_type)


def get_prediction_reward(supervised_network, episode, question_type):
    tr_cols = YOKED_TRAINING_COLS + MOUSE_CONTROL_COLS
    if question_type == "mass":
        class_columns = MASS_CLASS_COLS
    else:
        class_columns = FORCE_CLASS_COLS
    episode, answer = episode
    trial_data = make_dataframe_from_episode(episode, answer, class_columns)
    test_loader = prepare_dataset([trial_data], training_columns=tr_cols,
                                               class_columns=class_columns)
    accuracy = evaluate(supervised_network, test_loader)
    if accuracy == 100.:
        return 1.
    return -1.


def make_dataframe_from_episode(episode, answer, class_columns):
    trial_data = pd.DataFrame()
    for object_id in ["o1", "o2", "o3", "o4"]:
        for attr in ["x", "y", "vx", "vy"]:
            trial_data[object_id+"."+attr] = episode[object_id][attr]

    for cl in class_columns:
        trial_data[cl] = (answer == cl)
    for i, obj_id in enumerate(MOUSE_CONTROL_COLS): 
        trial_data[obj_id] = (episode["co"] == i)
    trial_data["mouseX"] = episode["mouse"]["x"]
    trial_data["mouseY"] = episode["mouse"]["y"]
    trial_data["mouse.vx"] = episode["mouse"]["vx"]
    trial_data["mouse.vy"] = episode["mouse"]["vy"]
    return trial_data

def supervised_learning(model, error, optimizer, supervised_data, question_type):

    tr_cols = YOKED_TRAINING_COLS + MOUSE_CONTROL_COLS
    if question_type == "mass":
        class_columns = MASS_CLASS_COLS
    else:
        class_columns = FORCE_CLASS_COLS

    dataset = []
    for episode, answer in supervised_data:
        trial_data = make_dataframe_from_episode(episode, answer, class_columns)
        dataset.append(trial_data)

    train_dataset, val_dataset = train_test_split(dataset, random_state=42, shuffle=True)
    train_loader, val_loader = prepare_dataset([train_dataset, val_dataset], training_columns=tr_cols,
                                               class_columns=class_columns)
    _, _, supervised_network = training_loop(model, optimizer, error, train_loader, val_loader,
                                             num_epochs=10, print_stats_per_epoch=True,
                                             seq_start=None, seq_end=None, step_size=None,
                                             patience=np.inf)

def learn(valueNetwork, target_network, sampled_experience, discountFactor, optimizer):

    acc_loss = 0
    loss = 0
    states, actions, rewards, new_states, dones = [], [], [], [], []
    for (state, action, reward, new_state, done) in sampled_experience:
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        new_states.append(new_state)
        dones.append(done)

    target_network = target_network.cuda()
    valueNetwork = valueNetwork.cuda()
    states = torch.cat(states, dim=0).cuda()
    new_states = torch.cat(new_states, dim=0).cuda()
    rewards = torch.tensor(rewards).cuda()
    dones = torch.tensor(dones).cuda()

    targets = rewards + discountFactor * target_network(new_states).max(dim=1)[0]
    targets = torch.where(dones, rewards, targets)
    targets = Variable(targets.clone().detach())

    outputs = valueNetwork(states)[:, np.arange(len(states[0])), actions]
    error = nn.SmoothL1Loss()

    loss = error(outputs, targets) / len(sampled_experience)
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


def validate(valueNetwork, mass_answers={}, force_answers={}, val_cond=()):
    print("----------------------------VALIDATION START-----------------------------------------")


    if len(mass_answers) > 0:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, mass_answers=mass_answers)
        question_type = 'mass'
    else:
        env = physic_env(val_cond, None, None, (3., 2.), 1, ig_mode=0, prior=None,
                         reward_stop=None, force_answers=force_answers)
        question_type = 'force'

    if os.path.exists("replays.h5"):
        os.remove("replays.h5")

    epsilon = 0.01
    correct_answers = 0

    answers = []
    ground_truth = []
    for episodeNumber in tqdm(range(len(val_cond))):
        done = False
        valueNetwork.reset_hidden_states()
        object_in_control = 0
        is_answer_correct = False

        object_A_in_control = False
        object_B_in_control = False
        frame = 0

        state = env.reset(True)
        answer = env.get_force_true_answer()
        answer = np.array(["attract", "repel", "none"]) == answer
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
                if question_type == "force":
                    print(force_answers[greedy_action], env.get_force_true_answer())
                elif question_type == "mass":
                    print(mass_answers[greedy_action], env.get_mass_true_answer())
            if info["control"]  == 1 and not object_A_in_control:
                reward += 1
                object_in_control += 0.5
                object_A_in_control = True
            if info["control"]  == 2 and not object_B_in_control:
                reward += 1
                object_in_control += 0.5
                object_B_in_control = True

            state = new_state
        # print(actions)
        data = env.step_data()
        trial_data = pd.DataFrame()

        for object_id in ["o1", "o2", "o3", "o4"]:
            for attr in ["x", "y", "vx", "vy"]:
                trial_data[object_id+"."+attr] = data[object_id][attr]

        trial_data["ground_truth"] = env.get_force_true_answer()
        trial_data["answer"] = force_answers[greedy_action]
        trial_data["mouseX"] = data["mouse"]["x"]
        trial_data["mouseY"] = data["mouse"]["y"]
        trial_data["mouse.vx"] = data["mouse"]["vx"]
        trial_data["mouse.vy"] = data["mouse"]["vy"]
        trial_data["idControlledObject"] = ["none" if obj == 0 else "object"+str(obj) for obj in data["co"]]
        trial_data.to_hdf("replays.h5", key="episode_"+str(episodeNumber))

        print("Correct?", is_answer_correct, "Control", object_in_control, "done @", frame)
    print("Correct perc", correct_answers / len(val_cond))
    print("----------------------------VALIDATION END-----------------------------------------")
