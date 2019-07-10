#!/usr/bin/env python3
# encoding utf-8

from copy import deepcopy
import argparse
import numpy as np
import torch.optim as optim
from torch import multiprocessing as mp

from SharedAdam import SharedAdam
from models import ValueNetwork
from Worker import train, validate, saveModelNetwork

import os
from action_coding import mass_answers, force_answers
from simulator.config import generate_every_world_configuration
from generate_passive_simulations import get_configuration_answer
from simulator.config import generate_cond
from sklearn.model_selection import train_test_split
import torch


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--question_type', type=str, default="force")

    args=parser.parse_args()
    if args.question_type == "mass":
        mass_answers = mass_answers
        force_answers = {}
    else:
        mass_answers = {}
        force_answers = force_answers

    counter = 0
    torch.backends.cudnn.benchmark = False
    net_params = {"input_dim":27, "hidden_dim":100, "n_layers":2, "output_dim":9, "dropout":0.5}
    value_network = ValueNetwork(**net_params).cuda()
    target_value_network = deepcopy(value_network)
    optimizer = optim.Adam(value_network.parameters(), lr=1e-2)

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
    N_WORLDS = 3000
    episodes_per_agent = 1000

    VALIDATION_EPISODES = 10
    print("episodes_per_agent", episodes_per_agent)
    i = 0

    torch.manual_seed(0)
    np.random.seed(0)
    repeated_train_indices = np.random.choice(train_indices, N_WORLDS, replace=True)
    train_cond = generate_cond(every_conf[repeated_train_indices])

    repeated_val_indices = np.random.choice(val_indices, VALIDATION_EPISODES, replace=False)
    val_cond = generate_cond(every_conf[repeated_val_indices])

    for i in range(N_WORLDS//episodes_per_agent):

        agent_cond = train_cond[i*episodes_per_agent: (i+1) * episodes_per_agent]
        startingEpisode = i * episodes_per_agent

        trainingArgs = (value_network, target_value_network, optimizer, counter,
                       episodes_per_agent, discountFactor, startingEpisode,
                       mass_answers, force_answers, agent_cond)
        train(*trainingArgs)

        valArgs = (value_network, mass_answers, force_answers, val_cond)
        p = mp.Process(target=validate, args=valArgs)
        p.start()
        p.join()

        i += 1
