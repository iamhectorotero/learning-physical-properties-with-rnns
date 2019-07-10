#!/usr/bin/env python3
# encoding utf-8

from copy import deepcopy
import argparse
import numpy as np
from torch import multiprocessing as mp
import torch

from SharedAdam import SharedAdam
from models import ValueNetwork
from Worker import train, validate, saveModelNetwork

import os
from action_coding import mass_answers, force_answers
from simulator.config import generate_every_world_configuration
from generate_passive_simulations import get_configuration_answer
from sklearn.model_selection import train_test_split


if __name__ == "__main__" :

    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_processes', type=int, default=12)
    parser.add_argument('--question_type', type=str)
    parser.add_argument('--model_path', type=str, default="500000_model")

    args=parser.parse_args()
    if args.question_type == "mass":
        mass_answers = mass_answers
        force_answers = {}
    else:
        mass_answers = {}
        force_answers = force_answers

    net_params = {"input_dim":26, "hidden_dim":25, "n_layers":4, "output_dim":9, "dropout":0.5}
    value_network = ValueNetwork(**net_params)
    value_network.load_state_dict(torch.load(args.model_path))
    value_network.eval()

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

    episodes_per_agent = 100
    print("episodes_per_agent", episodes_per_agent)

    startingEpisode = 0
    idx = 0
    valArgs = (idx, value_network, episodes_per_agent, mass_answers, 
              force_answers, every_conf, val_indices)
    p = mp.Process(target=validate, args=valArgs)
    p.start()
    p.join()

