#!/usr/bin/env python3
# encoding utf-8

from copy import deepcopy
import argparse
import numpy as np
import torch.optim as optim
from torch import multiprocessing as mp

from SharedAdam import SharedAdam
from models import ValueNetwork, NonRecurrentValueNetwork, Policy
from RecurrentWorker import train, validate, saveModelNetwork
from PolicyGradients import train_pg, validate_pg

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
    parser.add_argument('--question_type', type=str, default="mass")

    args=parser.parse_args()
    if args.question_type == "mass":
        mass_answers = mass_answers
        force_answers = {}
    else:
        mass_answers = {}
        force_answers = force_answers

    # mp.set_start_method('spawn')
    counter = mp.Value('i', 0)
    lock = mp.Lock()

    torch.backends.cudnn.benchmark = False
    net_params = {"input_dim":29, "hidden_dim":30, "n_layers":2, "output_dim":9, "dropout":0.5}
    # policy_network = Policy(**net_params)
    value_network = ValueNetwork(**net_params).cuda()
    target_value_network = deepcopy(value_network)
    optimizer = SharedAdam(value_network.parameters(), lr=1e-3)
    # optimizer = optim.Adam(policy_network.parameters(), lr=1e-7)

    value_network.share_memory()
    target_value_network.share_memory()
    optimizer.share_memory()

    discountFactor = 0.9

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
    episodes_per_agent = 250
    val_episodes_per_agent = 11

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
    while True:
        processes = []
        for idx in range(0, args.num_processes):

            agent_cond = train_cond[:episodes_per_agent]
            train_cond = train_cond[episodes_per_agent:]

            startingEpisode = i * episodes_per_agent

            trainingArgs = (value_network, target_value_network, optimizer, counter,
                           episodes_per_agent, discountFactor, startingEpisode,
                           mass_answers, force_answers, agent_cond, idx, lock)
            train(*trainingArgs)
            # p = mp.Process(target=train, args=trainingArgs)
            """trainingArgs = (policy_network, optimizer, counter,
                           episodes_per_agent, discountFactor, startingEpisode,
                           mass_answers, force_answers, agent_cond, idx, lock)
            p = mp.Process(target=train_pg, args=trainingArgs)"""
            # p.start()
            # processes.append(p)
        for p in processes:
            p.join()

        i += 1

        agent_cond = val_cond[:val_episodes_per_agent]
        val_cond = val_cond[val_episodes_per_agent:]
        valArgs = (value_network, mass_answers, force_answers, agent_cond)
        # valArgs = (policy_network, mass_answers, force_answers, val_cond)
        # p = mp.Process(target=validate, args=valArgs)
        # p.start()
        # p.join()
        validate(*valArgs)
        if len(train_cond) == 0:
            exit()

