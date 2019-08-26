#!/usr/bin/env python3
# encoding utf-8

from copy import deepcopy
import argparse
import numpy as np
import torch.optim as optim
from torch import multiprocessing as mp
import pandas as pd

from toddler.models import ValueNetwork
from isaac.models import ComplexRNNModel
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
    net_params = {"input_dim":17, "hidden_dim":25, "n_layers":4, "output_dim":9, "dropout":0.0}
    value_network = ValueNetwork(**net_params).cuda()
    optimizer = optim.Adam(value_network.parameters(), lr=5e-4)

    net_params = {"input_dim":17, "hidden_dim":25, "n_layers":4, "output_dim":3, "dropout":0.0}
    yoked_network = ComplexRNNModel(**net_params).cuda()
    yoked_optimizer = optim.Adam(yoked_network.parameters(), lr=0.005)

    discountFactor = 0.99

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
    N_WORLDS = 100000
    worlds_per_agent = N_WORLDS // args.num_processes
    episodes_per_agent = 1000
    val_episodes_per_agent = 10

    n_iterations = N_WORLDS // episodes_per_agent + 1
    VALIDATION_EPISODES = val_episodes_per_agent*n_iterations
    print("episodes_per_agent", episodes_per_agent)
    i = 0

    torch.manual_seed(0)
    np.random.seed(0)
    repeated_train_indices = np.random.choice(train_indices, N_WORLDS, replace=True)
    train_cond = generate_cond(every_conf[repeated_train_indices])

    repeated_val_indices = np.random.choice(val_indices, VALIDATION_EPISODES, replace=True)
    val_cond = generate_cond(every_conf[repeated_val_indices])

    experience_replay = ()
    agent_answers = ()
    training_data = {"question_loss":[], "value_loss":[], "control": [], "episode_length": [], "correct_answer": []}

    n_bodies = 2
    action_repeat = 1
    MAX_STARTING_SPEED = 10
    MIN_STARTING_SPEED = 3
    total_steps = 0

    for i in range(n_iterations):
        starting_puck_speed = MIN_STARTING_SPEED + (MAX_STARTING_SPEED - MIN_STARTING_SPEED) * i / (n_iterations - 1)
        agent_cond = train_cond[:episodes_per_agent]
        train_cond = train_cond[episodes_per_agent:]

        startingEpisode = i * episodes_per_agent

        trainingArgs = (value_network, optimizer, episodes_per_agent, discountFactor,
                        startingEpisode, mass_answers, force_answers, agent_cond,
                        experience_replay, agent_answers, n_bodies, training_data,
                        starting_puck_speed, total_steps, yoked_network, yoked_optimizer)
        experience_replay, agent_answers, training_data, total_steps = train(*trainingArgs)

        df = pd.DataFrame.from_dict(training_data)
        df.to_hdf("training_data.h5", key="training_data")

        agent_cond = val_cond[:val_episodes_per_agent]
        val_cond = val_cond[val_episodes_per_agent:]

        for cond in val_cond:
            cond['lf'] = [[0.0, 0.0], [0.0, 0.0]]
            cond['lf'] = [[0.0, 0.0], [0.0, 0.0]]
            vs = np.random.uniform(-0.0, 0.0, (n_bodies, 2))
            cond['svs'] = [{"x": vs[i][0], "y": vs[i][1]} for i in range(n_bodies)]

        valArgs = (value_network, mass_answers, force_answers, agent_cond, n_bodies, action_repeat,
                   "replays.h5")
        validate(*valArgs)

        if len(train_cond) == 0:
            break

