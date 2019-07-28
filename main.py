#!/usr/bin/env python3
# encoding utf-8

from copy import deepcopy
import argparse
import numpy as np
import torch.optim as optim
from torch import multiprocessing as mp
import pandas as pd

from toddler.models import ValueNetwork, NonRecurrentValueNetwork, Policy
from toddler.RecurrentWorker import train, saveModelNetwork
from toddler.validate import validate

import os
from toddler.action_coding import mass_answers, force_answers
from simulator.config import generate_every_world_configuration
from generate_passive_simulations import get_configuration_answer
from simulator.config import generate_cond
from sklearn.model_selection import train_test_split
import torch


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--question_type', type=str, default="mass")

    args=parser.parse_args()
    if args.question_type == "mass":
        mass_answers = mass_answers
        force_answers = {}
    else:
        mass_answers = {}
        force_answers = force_answers

    torch.backends.cudnn.benchmark = False
    net_params = {"input_dim":10, "hidden_dim":40, "n_layers":4, "output_dim":6, "dropout":0.0}
    value_network = ValueNetwork(**net_params).cuda()
    optimizer = optim.Adam(value_network.parameters(), lr=5e-4)

    discountFactor = 0.95

    every_conf = generate_every_world_configuration()
    every_world_answer = np.array(list(map(get_configuration_answer, every_conf)))
    n_configurations = len(every_conf)

    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    all_indices = np.arange(n_configurations)
    train_indices, not_train_indices = train_test_split(all_indices, train_size=train_size,
                                                        random_state=0, stratify=every_world_answer)
    val_indices, test_indices = train_test_split(not_train_indices, train_size=0.5,                
                                                 random_state=0,
                                                 stratify=every_world_answer[not_train_indices])

    TOTAL_STEPS = int(32e6)
    N_WORLDS = 50000
    worlds_per_agent = N_WORLDS // args.num_processes
    episodes_per_agent = 10
    val_episodes_per_agent = 10

    VALIDATION_EPISODES = val_episodes_per_agent *(N_WORLDS // 250)
    print("episodes_per_agent", episodes_per_agent)
    i = 0

    torch.manual_seed(0)
    np.random.seed(0)
    repeated_train_indices = np.random.choice(train_indices, N_WORLDS, replace=True)
    train_cond = generate_cond(every_conf[repeated_train_indices])

    repeated_val_indices = np.random.choice(val_indices, VALIDATION_EPISODES, replace=True)
    val_cond = generate_cond(every_conf[repeated_val_indices])

    i = 0
    experience_replay = ()
    agent_answers = ()
    training_data = {"loss":[], "control": [], "episode_length": []}

    n_bodies = 1
    action_repeat = 1

    while True:
        agent_cond = train_cond[:episodes_per_agent]
        train_cond = train_cond[episodes_per_agent:]

        startingEpisode = i * episodes_per_agent

        trainingArgs = (value_network, optimizer, episodes_per_agent, discountFactor,
                        startingEpisode, mass_answers, force_answers, agent_cond,
                        experience_replay, agent_answers, n_bodies, training_data)
        experience_replay, agent_answers, training_data = train(*trainingArgs)

        df = pd.DataFrame.from_dict(training_data)
        df.to_hdf("training_data.h5", key="training_data")

        agent_cond = val_cond[:val_episodes_per_agent]
        val_cond = val_cond[val_episodes_per_agent:]
        valArgs = (value_network, mass_answers, force_answers, agent_cond, n_bodies, action_repeat,
                   "replays.h5")
        validate(*valArgs)

        i += 1

        if len(train_cond) == 0:
            break

