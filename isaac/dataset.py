import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.utils.data
from tqdm import tqdm

from .constants import BASIC_TRAINING_COLS

def read_dataset(path, first_n_trials=np.inf):
    dataset = []
    trial_count = 0

    while trial_count < first_n_trials:
        try:
            dataset.append(pd.read_hdf(path, key="trial_"+str(trial_count), dtype=np.float32))
            trial_count += 1
        except KeyError:
            break
    
    return dataset


def get_sliding_windows_for_all_trials(trials, window_size=4):

    old_shape = trials.shape
    
    window_starting_points = old_shape[1] - window_size
    new_shape = (old_shape[0], window_starting_points, window_size, old_shape[-1])
    
    bytes_between_attributes = trials.strides[2]
    bytes_between_rows_in_window = trials.strides[1]
    bytes_between_window_starts = trials.strides[1]
    bytes_between_trials = trials.strides[0]
    
    new_strides = (bytes_between_trials, 
                   bytes_between_window_starts, 
                   bytes_between_rows_in_window, 
                   bytes_between_attributes)
    
    return as_strided(trials, new_shape, new_strides)


def prepare_dataset(dataset, class_columns, batch_size=640, normalise_data=False, test_size=0.2, equiprobable_training_classes=False,
                    transforms=(), sliding_window_size=1, training_columns=BASIC_TRAINING_COLS):

    X = []
    Y = []

    for trial in tqdm(dataset):
        training_cols = trial[training_columns]
        
        for t in transforms:
            t(training_cols)
            
        X.append(np.array(training_cols).astype(np.float32))
        Y.append(np.argmax(np.array(trial[class_columns].iloc[0])))

    X = np.array(X)
    if sliding_window_size > 1:
        X = get_sliding_windows_for_all_trials(X, sliding_window_size)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])
    Y = np.array(Y)

    X_train, X_val, Y_train, Y_val = split_data_in_train_and_test(X, Y, test_size, equiprobable_training_classes)
    
    scaler = None
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


def prepare_test_dataset(dataset, class_columns, batch_size=640, transforms=(), sliding_window_size=1,
                         scaler=None, training_columns=BASIC_TRAINING_COLS):

    X = []
    Y = []

    for trial in tqdm(dataset):
        training_cols = trial[training_columns]

        for t in transforms:
            t(training_cols)

        X.append(np.array(training_cols).astype(np.float32))
        Y.append(np.argmax(np.array(trial[class_columns].iloc[0])))

    X = np.array(X)
    if sliding_window_size > 1:
        X = get_sliding_windows_for_all_trials(X, sliding_window_size)
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])
    Y = np.array(Y)

    if scaler:
        X_test = normalise(X, scaler, fit_scaler=False)

    X_test = torch.from_numpy(X).cuda()
    Y_test = torch.from_numpy(Y).type(torch.LongTensor).cuda()

    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


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
        
