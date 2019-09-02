import numpy as np


#TODO: change name to class_counts_and_majority_class_proportion

def class_proportions(dataset_loader):
    Y = []
    for x,y in dataset_loader:

        Y.extend(list(y))

    counts = np.unique(Y, return_counts=True)[1]
    print(counts)
    majority_class_proportion = np.max(counts) / np.sum(counts)
    print("Majority class: ", majority_class_proportion)

    return counts, majority_class_proportion
