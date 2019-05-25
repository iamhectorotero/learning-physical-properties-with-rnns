import pandas as pd
import numpy as np
from constants import BASIC_TRAINING_COLS
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data


def read_dataset(path):
    dataset = []
    trial_count = 0

    while True:
        try:
            dataset.append(pd.read_hdf(path, key="trial_"+str(trial_count), dtype=np.float32))
            trial_count += 1
        except KeyError:
            break
    
    return dataset

def prepare_dataset(dataset, class_columns, normalise=False):

    X = []
    Y = []

    for trial in dataset:
        X.append(np.array(trial[BASIC_TRAINING_COLS]).astype(np.float32))
        Y.append(np.argmax(np.array(trial[class_columns].iloc[0])))

    X = np.array(X)
    Y = np.array(Y)
    
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    if normalise:
        attr_means = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        attr_std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
        X_train = (X_train - attr_means) / attr_std
        X_val = (X_val - attr_means) / attr_std
        
    X_train = torch.from_numpy(X_train).cuda()
    X_val = torch.from_numpy(X_val).cuda()
    Y_train = torch.from_numpy(Y_train).type(torch.LongTensor).cuda()
    Y_val = torch.from_numpy(Y_val).type(torch.LongTensor).cuda()
    
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=640, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader