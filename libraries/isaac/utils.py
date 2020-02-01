from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
import os
from shutil import rmtree

from .constants import BASIC_TRAINING_COLS


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = ''
        else:
            title = ''

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        pass
        #print('Confusion matrix, without normalization')

    # print(cm)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))

    vmin, vmax = None, None
    if normalize:
        vmin = 0
        vmax = 100

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    # ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '{0:.0f}%' if normalize else '{}'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, fmt.format(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=25, fontweight='bold')
    return ax

def plot_timeseries(timeseries, labels, xlabel=None, ylabel=None):
    plt.figure(figsize=(11, 7))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    for length, ts_item in zip(labels, timeseries):
        print(ts_item)
        plt.errorbar(np.arange(timeseries.shape[-1]), ts_item.mean(axis=0), yerr=ts_item.std(axis=0), 
                     label=str(length))
    plt.legend()


def get_cuda_device_if_available():
    """Returns:
        device: A torch.device object with the first cuda device (cuda:0) if any are available.
        Otherwise it returns the cpu device.
    """

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"

    return torch.device(device)


def create_directory(path, delete_if_exists=False):
    """Creates a directory in the specified path.
    Args:
        path (str): the path where the directory will be created (the name of the directory is included
        in the path). If some of the directories on the way to the final one are missing, they
        will also be created.

        delete_if_exists (bool): If true and the directory exists, it will be first fully deleted (rm -rf)
        and then created empty.
    """

    path_exists = os.path.exists(path)

    if not path_exists:
        os.makedirs(path, exist_ok=True)

    if path_exists and delete_if_exists:
        rmtree(path)
        os.makedirs(path, exist_ok=True)

