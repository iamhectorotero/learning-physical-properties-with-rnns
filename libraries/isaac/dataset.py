import os
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import OrderedDict
import torch
import torch.utils.data
from tqdm import tqdm

from .constants import BASIC_TRAINING_COLS, YOKED_TRAINING_COLS, MASS_CLASS_COLS, FORCE_CLASS_COLS

def read_dataset(path, n_trials=None, shuffle=False, seed=0, cols=None, add_class_columns=True):
    """Reads a local hdf/h5 file consisting of one or more DataFrames whose keys are of the form
    trial_X where X is a number.

    Args:
        path: the file path.
        n_trials: The number of DataFrames to be read. If larger than the total number of DataFrames
        stored, it will default to this value. If lower, a random subset will be sampled.
        shuffle: (boolean) whether to read the trials in order or shuffle them.
        seed: a seed for Numpy to control in which order DataFrames are read. If shuffle is False,
        this argument will be ignored.
        cols: if specified, only these columns  will be read from the DataFrames. If duplicated, they will only be read once.
        add_class_columns: (default True) class columns (see constants MASS_CLASS_COLS and FORCE_CLASS_COLS) will be added to the list of cols (if those are specified).
    Returns:
        dataset: a list of Pandas DataFrames. One per HDF key found in the file path.
    """

    np.random.seed(seed)

    with pd.HDFStore(path, mode="r") as hdf:
        keys = dir(hdf.root)[124:]

        get_trial_id = lambda x: int(x.split("_")[1])
        keys = sorted(keys, key=get_trial_id)

        if n_trials is None or n_trials > len(keys):
            n_trials = len(keys)

        if shuffle:
            trials = np.random.choice(keys, size=n_trials, replace=False)
        else:
            trials = keys[:n_trials]

        if cols is not None:
            cols = list(cols)
            if add_class_columns:
                cols += MASS_CLASS_COLS
                cols += FORCE_CLASS_COLS
            cols = list(OrderedDict.fromkeys(cols))
            dataset = [hdf[trial_i][cols] for trial_i in tqdm(trials)]
        else:
            dataset = [hdf[trial_i] for trial_i in tqdm(trials)]
        
    return dataset


def are_classes_one_hot_encoded(class_values):
    """Checks if a vector is one-hot encoded
    Args:
        class_values: a 1D vector.
    Returns:
        boolean: True if the vector is one-hot encoded. False otherwise"""

    assert len(np.array(class_values).shape) == 1
    unique_values, counts = np.unique(class_values, return_counts=True)
    return np.array_equal(unique_values, [0, 1]) and counts[1] == 1


def prepare_dataset(datasets, class_columns, multiclass=False, batch_size=640, normalise_data=False, scaler=None,
                    transforms=(), sliding_window_size=1, training_columns=BASIC_TRAINING_COLS,
                    categorical_columns=(), normalisation_cols=(), device=torch.device("cpu")):

    """
    Args:
    datasets: list of datasets to which the same transformations will be applied. Inside a dataset,
    all trials must be of the same length.

    class_columns: iterable. If multiclass is False, then it represents a single class columns. If multiclass is True,
    then it represents a list of classes for training a multiple-branch network.

    multiclass: boolean. Modifies the behavior of class_columns as above.

    batch_size: integer. batch size of every dataset loader

    normalise_data: boolean. whether to apply a StandardScaler normalisation to the data. if datasets contains more than
    one dataset, it will be fitted in the first dataset and applied to the rest.

    scaler: if None and normalise_data=True, it will be fitted to the data. If a scaler previously fitted, then it will
    be used to transform every dataset.

    transforms: iterable. Functions to be applied to each trial of every dataset. Each transform receives a Pandas DataFrame and should modify it inplace.

    sliding_window_size: integer. If larger than 1, the returned dataset will consist of windows of the indicated size.

    training_columns: iterable. These columns will be extracted from the dataset so that they can be used to predict.

    categorical_columns: iterable. These columns (which must exist in every trial of every dataset)
    will not be normalised even if normalise_data is True. Class columns will not be normalised.

    normalisation_cols: iterable. If not specified, it defaults to all training columns. If
    specified, only columns in this list and not in the categorical_columns list will be normalised.
    Execution will fail if a column is in both "categorical_columns" and "normalisation_cols".

    device: the datasets will be sent to this torch.device. Defaults to cpu.

    Returns:
        loaders: if len(datasets) > 1, a list of loaders corresponding to the list of datasets passed
        as arguments. If len(datasets) == 1, a single DatasetLoader will be returned.

        scaler: the fitted or used dataset. None, if no scaler was passed and no normalisation
        took place.
    """

    if len(normalisation_cols) == 0:
        normalisation_cols = training_columns
    elif len(set(categorical_columns) & set(normalisation_cols)) > 0:
        raise ValueError("A column is listed both as categorical and as column to normalise")

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
                    class_values = np.array(trial[class_i_columns].iloc[0])
                    if not are_classes_one_hot_encoded(class_values):
                        raise ValueError("Classes are not one-hot encoded")
                    y.append(np.argmax(class_values))
                Y.append(y)
            else:
                class_values = np.array(trial[class_columns].iloc[0])
                if not are_classes_one_hot_encoded(class_values):
                    raise ValueError("Classes are not one-hot encoded")
                Y.append(np.argmax(class_values))

        X = np.array(X)
        """if sliding_window_size > 1:
            X = get_sliding_windows_for_all_trials(X, sliding_window_size)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])"""
        Y = np.array(Y)

        if normalise_data:
            if scaler is None:
                scaler = StandardScaler()
                X = normalise(X, scaler, fit_scaler=True, columns_to_normalise_bool_index=columns_to_normalise_bool_index)
            else:
                X = normalise(X, scaler, fit_scaler=False, columns_to_normalise_bool_index=columns_to_normalise_bool_index)

        X = torch.from_numpy(X).to(device=device)
        Y = torch.from_numpy(Y).type(torch.LongTensor).to(device=device)

        tensor_dataset = torch.utils.data.TensorDataset(X, Y)
        data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
        loaders.append(data_loader)

    if len(datasets) == 1:
        return loaders[0], scaler

    return loaders, scaler


def normalise(X, scaler, fit_scaler=True, columns_to_normalise_bool_index=None):
    """Applies the scaler to the numpy array data.
    Args:
        X: a 3D NumPy array (batch, sequence, features).

        scaler: a scikit-learn scaler. E.g. StandardScaler

        fit_scaler: (default: True) whether to fit the scaler to the provided data. If False, the
        scaler must have been previously fitted.

        columns_to_normalise_bool_index: boolean index of length equal to the features. If provided,
        columns whose position in the index are False won't be normalised. Uses: not normalising
        categorical features.
    Returns:
        X: A normalised version of the X passed as parameter.
    """

    original_shape = X.shape
    X = X.reshape(-1, original_shape[-1])

    if columns_to_normalise_bool_index is None:
        columns_to_normalise_bool_index = np.full(original_shape[-1], True)

    if fit_scaler:
        scaler.fit(X[:, columns_to_normalise_bool_index])

    X[:, columns_to_normalise_bool_index] =  scaler.transform(X[:, columns_to_normalise_bool_index])

    return X.reshape(original_shape)


"""
# These are unused and thus untested methods. Use at your own risk.

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
        
    return subsequences"""
