import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import deepcopy
from isaac.utils import get_cuda_device_if_available, create_directory
from isaac.dataset import read_dataset, prepare_dataset
from isaac.utils import plot_confusion_matrix
from isaac.constants import BASIC_TRAINING_COLS, FORCE_CLASS_COLS, MASS_CLASS_COLS
from isaac.training import training_loop
from isaac.models import MultiBranchModel, initialise_model
from isaac.evaluation import get_best_model_and_its_accuracy


device = get_cuda_device_if_available()
print(device)

data_directory = "large_multibranch/"
create_directory(data_directory)
create_directory("models")
create_directory("scalers")

BATCH_SIZE = 64
EPOCHS = 50
NORMALISE_DATA = True
STEP_SIZE = 3
SEQ_END = 2700

INPUT_DIM = len(BASIC_TRAINING_COLS)    # input dimension
HIDDEN_DIM = 90  # hidden layer dimension
N_LAYERS = 8     # number of hidden layers
OUTPUT_DIM = 3   # output dimension
DROPOUT = 0.5

network_params = (INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, DROPOUT)

train_trials = read_dataset("data/train_passive_trials.h5")
val_trials = read_dataset("data/val_passive_trials.h5")

N_MODELS = 3

stats_dfs = []
loaders, scaler = prepare_dataset([train_trials, val_trials], 
                                  class_columns=[list(MASS_CLASS_COLS), list(FORCE_CLASS_COLS)], 
                                  multiclass=True,
                                  batch_size=BATCH_SIZE, normalise_data=NORMALISE_DATA,
                                  device=device)


for seed in range(N_MODELS):
    df = pd.DataFrame()

    model, error, optimizer = initialise_model(network_params, lr=0.001, seed=seed, device=device, arch=MultiBranchModel)
    epoch_losses, epoch_accuracies, [best_mass_model, best_force_model] = training_loop(model, optimizer, 
                                                                                        error,
                                                                                        loaders[0], loaders[1], 
                                                                                        EPOCHS, seq_end=SEQ_END,
                                                                                        step_size=STEP_SIZE,
                                                                                        multibranch=True)

    torch.save(best_mass_model.state_dict(), "models/best_mass_model_seed_%d.pt" % seed)
    torch.save(best_force_model.state_dict(), "models/best_force_model_seed_%d.pt" % seed)
    
    
    train_accuracies = np.array(epoch_accuracies[0])
    val_accuracies = np.array(epoch_accuracies[1]) 
    
    df["Epoch"] = np.arange(EPOCHS)
    df["Mass Loss"] = epoch_losses[:, 0]
    df["Force Loss"] = epoch_losses[:, 1]
    df["Mass Train Accuracy"] = train_accuracies[:, 0]
    df["Mass Val Accuracy"] = val_accuracies[:, 0]
    df["Force Train Accuracy"] = train_accuracies[:, 1]
    df["Force Val Accuracy"] = val_accuracies[:,1]
    df["seed"] = str(seed)
    stats_dfs.append(df)
        
stats = pd.concat(stats_dfs)
stats.to_hdf(data_directory+"stats.h5", key="stats")

joblib.dump(scaler, "scalers/passive_dual_scaler.sk")
