import numpy as np


def class_proportions(dataset_loader):
    Y = []
    for x,y in dataset_loader:

        Y.extend(list(y))

    counts = np.unique(Y, return_counts=True)[1]
    print(counts)
    print("Majority class: ", np.max(counts) / np.sum(counts))