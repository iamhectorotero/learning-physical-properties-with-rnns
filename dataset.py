import pandas as pd
import numpy as np
from constants import BASIC_TRAINING_COLS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

def prepare_dataset(dataset, class_columns, batch_size=640, normalise_data=False, test_size=0.2, equiprobable_training_classes=True):

    X = []
    Y = []

    for trial in dataset:
        X.append(np.array(trial[BASIC_TRAINING_COLS]).astype(np.float32))
        Y.append(np.argmax(np.array(trial[class_columns].iloc[0])))

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_val, Y_train, Y_val = split_data_in_train_and_test(X, Y, test_size, equiprobable_training_classes)
    
    if normalise_data:
        scaler = StandardScaler()
        X_train = normalise(X_train, scaler, fit_scaler=True)
        X_val = normalise(X_val, scaler, fit_scaler=False) 
        
    X_train = torch.from_numpy(X_train).cuda()
    X_val = torch.from_numpy(X_val).cuda()
    Y_train = torch.from_numpy(Y_train).type(torch.LongTensor).cuda()
    Y_val = torch.from_numpy(Y_val).type(torch.LongTensor).cuda()
    
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, scaler

def normalise(X, scaler, fit_scaler=True):
    original_shape = X.shape
    X = X.reshape(-1, original_shape[-1])

    if fit_scaler:
        scaler.fit(X)
        
    return scaler.transform(X).reshape(original_shape)

def split_data_in_train_and_test(X, Y, test_size, equiprobable_training_classes=True):
    
    if equiprobable_training_classes:
        classes, counts = np.unique(Y, return_counts=True)
        min_class_count = int(min(counts) * (1 - test_size))

        train_indices = np.concatenate([np.where(Y == class_i)[0][:min_class_count] for class_i in classes])
        test_indices = list(set(np.arange(len(Y))) - set(train_indices))
        
        return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
    
    return train_test_split(X, Y, test_size=test_size, random_state=42)
        