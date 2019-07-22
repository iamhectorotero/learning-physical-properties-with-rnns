#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
config.py
"""
import numpy as np
import os
import json
# --- Set constants ---
TARGET_FPS = 60
TIME_STEP = 1.0 / TARGET_FPS
WIDTH, HEIGHT = 6, 4
BALL_RADIUS = 0.25
BORDER = 0.20
SIGMA = np.array([[0.2758276,0],[0,0.6542066]])

# hyper-parameter
T = 1
TIMEOUT = 2700
ig_mode = 1
state_dim = T*16
n_actions = 645
nn_h1 = 150
nn_h2 = 250
nn_h3 = 450
epsilon = 0.5
epsilon_decay = 0.95
qlearning_gamma = 0.99 
init_mouse = (3, 2)
reward_stop = 0.95


# Functions for generating candidate settings
def transfer(n,x):
    b=[]
    while True:
        s=n//x
        y=n%x
        b=b+[y]
        if s==0:
            break
        n=s
    b.reverse()
    return b


def generate_force(force_possible):
    force_list = []
    for num in force_possible:
        num = np.array(num)
        force = np.zeros([4,4])
        force[1,0] = (num[0]-1)*3
        force[2,:2] = (num[1:3]-1)*3
        force[3,:3] = (num[3:]-1)*3
        force = force+force.T
        force_list.append(force)
    return force_list


def generate_possible(level, length):
    possible = []
    for i in range(level**length):
        possible.append([0]*(length-len(transfer(i,level)))+transfer(i,level))
    return possible

# Functions for generating initial conditions
def generate_cond(configurations, timeout=TIMEOUT):
    cond_list = []
    # locations:
    for mass_configuration, force_configuration in configurations:
        X = 5.1*np.random.rand(4) + BALL_RADIUS + BORDER
        Y = 3.1*np.random.rand(4) + BALL_RADIUS + BORDER
        VX = np.random.uniform(-2.5, 2.5, 4)
        VY = np.random.uniform(-2.5, 2.5, 4)

        cond_list.append({'sls':[{'x':X[0], 'y':Y[0]}, {'x':X[1], 'y':Y[1]}, {'x':X[2], 'y':Y[2]}, {'x':X[3], 'y':Y[3]}],
        'svs':[{'x':VX[0], 'y':VY[0]}, {'x':VX[1], 'y':VY[1]}, {'x':VX[2], 'y':VY[2]}, {'x':VX[3], 'y':VY[3]}],
        'lf': generate_force([force_configuration])[0].tolist(),
        'mass': (np.array(mass_configuration)).tolist(),
        'timeout': timeout
        })
    return cond_list


def load_cond(file_name, size):
    if os.path.exists(file_name):
        with open(file_name,'r') as f:
            cond_list = json.load(f)
    else:
        cond_list = generate_cond(size)
        with open(file_name,'w') as f:
            json.dump(cond_list,f)
    return cond_list


def cartesian_product(mass_all_configs, force_all_configs):
    all_configs = []
    for mass_config in mass_all_configs:
        for force_config in force_all_configs:
            all_configs.append((mass_config, force_config))

    return all_configs

def generate_every_world_configuration():
    
    mass_all_possible = [target_pucks_mass + [1, 1] for target_pucks_mass in [[1, 1], [1, 2], [2, 1]]]
    force_all_possible = np.array(generate_possible(3, 6))
    return np.array(cartesian_product(mass_all_possible, force_all_possible))

# train_cond = load_cond("train_cond"+str(TIMEOUT)+".json", 10000)
# test_cond = load_cond("test_cond"+str(TIMEOUT)+".json", 100)
# print("timeout", train_cond[0]['timeout'])
