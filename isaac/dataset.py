import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.utils.data
from tqdm import tqdm

from .constants import BASIC_TRAINING_COLS, YOKED_TRAINING_COLS, MASS_CLASS_COLS, FORCE_CLASS_COLS

def read_dataset(path, n_trials=None, seed=0, cols=None):
    np.random.seed(seed)

    with pd.HDFStore(path) as hdf:
        keys = dir(hdf.root)[124:]
        keys = sorted(keys)

        if n_trials is None or n_trials > len(keys):
            n_trials = len(keys)
        
        trials = np.random.choice(keys, size=n_trials, replace=False)
        
        if cols is not None:
            cols = list(cols)
            cols += MASS_CLASS_COLS
            cols += FORCE_CLASS_COLS
            dataset = [hdf[trial_i][cols] for trial_i in tqdm(trials)]
        else:
            dataset = [hdf[trial_i] for trial_i in tqdm(trials)]
        
    return dataset


def prepare_dataset(datasets, class_columns, multiclass=False, batch_size=640, normalise_data=False, scaler=None,
                    transforms=(), sliding_window_size=1, training_columns=BASIC_TRAINING_COLS,
                    categorical_columns=(), normalisation_cols=()):
    
    """
    datasets: list of datasets to which the same transformations will be applied.
    class_columns: iterable. If multiclass is False, then it represents a single class columns. If multiclass is True,
                   then it represents a list of classes for training a multiple-branch network.
    multiclass: boolean. Modifies the behavior of class_columns as above.
    batch_size: integer. batch size of every dataset loader
    normalise_data: boolean. whether to apply a StandardScaler normalisation to the data. if datasets contains more than
                    one dataset, it will be fitted in the first dataset and applied to the rest.
    scaler: if None and normalise_data=True, it will be fitted to the data. If a scaler previously fitted, then it will
            be used to transform every dataset.
    transforms: iterable. Functions to be applied to each trial of every dataset.
    sliding_window_size: integer. If larger than 1, the returned dataset will consist of windows of the indicated size.
    training_columns: iterable. These columns will be extracted from the dataset so that they can be used to predict.
    """
    
    if len(normalisation_cols) == 0:
        normalisation_cols = training_columns

    training_columns = list(training_columns)
    columns_to_normalise_bool_index = np.array([(col not in categorical_columns) and 
                                                (col in normalisation_cols) 
                                                for col in training_columns])
    class_columns = list(class_columns)

    loaders = []

    for dataset in datasets:
        X = []
        Y = []

        for trial in tqdm(dataset):
            training_cols = trial[training_columns]
            
            for t in transforms:
                t(training_cols)

            X.append(np.array(training_cols).astype(np.float32))

            if multiclass:
                y = []
                for class_i_columns in class_columns:
                    y.append(np.argmax(np.array(trial[class_i_columns].iloc[0])))
                Y.append(y)
            else:
                Y.append(np.argmax(np.array(trial[class_columns].iloc[0])))

        X = np.array(X)
        if sliding_window_size > 1:
            X = get_sliding_windows_for_all_trials(X, sliding_window_size)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])
        Y = np.array(Y)

        if normalise_data:
            if scaler is None:
                scaler = StandardScaler()
                X = normalise(X, scaler, fit_scaler=True, columns_to_normalise_bool_index=columns_to_normalise_bool_index)
            else:
                X = normalise(X, scaler, fit_scaler=False, columns_to_normalise_bool_index=columns_to_normalise_bool_index)

        X = torch.from_numpy(X).cuda()
        Y = torch.from_numpy(Y).type(torch.LongTensor).cuda()

        tensor_dataset = torch.utils.data.TensorDataset(X, Y)
        data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        loaders.append(data_loader)

    if len(datasets) == 1:
        return loaders[0], scaler

    return loaders, scaler


def normalise(X, scaler, fit_scaler=True, columns_to_normalise_bool_index=None):
    original_shape = X.shape
    X = X.reshape(-1, original_shape[-1])

    if columns_to_normalise_bool_index is None:
        columns_to_normalise_bool_index = np.full(original_shape[-1], True)

    if fit_scaler:
        scaler.fit(X[:, columns_to_normalise_bool_index])

    X[:, columns_to_normalise_bool_index] =  scaler.transform(X[:, columns_to_normalise_bool_index])

    return X.reshape(original_shape)


def split_data_in_train_and_test(X, Y, test_size, equiprobable_training_classes=True):

    if equiprobable_training_classes:
        classes, counts = np.unique(Y, return_counts=True)
        min_class_count = int(min(counts) * (1 - test_size))

        train_indices = np.concatenate([np.where(Y == class_i)[0][:min_class_count] for class_i in classes])
        test_indices = list(set(np.arange(len(Y))) - set(train_indices))
        
        return X[train_indices], X[test_indices], Y[train_indices], Y[test_indices]
    
    return train_test_split(X, Y, test_size=test_size, random_state=42, shuffle=False)


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

        
def sample_subsequences(trials, seq_size=600, n_samples=10):
    subsequences = []

    for trial in trials:
        starting_points = np.random.randint(0, len(trial)-seq_size, size=n_samples)
        
        for start in starting_points:
            subsequences.append(trial.iloc[start:start+seq_size])
    
    return subsequences


def subsample_sequences(trials, step_size=2):
    subsampled_sequences = []
    
    for trial in trials:
        subsampled_sequences.append(trial.iloc[::step_size])
        subsampled_sequences.append(trial.iloc[1::step_size])

    
    return subsampled_sequences

def split_in_subsequences(trials, seq_size):
    subsequences = []
    
    for trial in trials:
        length = len(trial)
        for i in range(length//seq_size):
            subsequences.append(trial.iloc[i*seq_size:(i+1)*seq_size])
        
    return subsequences
