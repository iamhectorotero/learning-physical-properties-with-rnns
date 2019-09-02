import numpy as np


#TODO: change name to class_counts_and_majority_class_proportion

def class_proportions(dataset_loader):
    """Checks the number of examples in each class and the majority class percentage.
    Args:
        dataset_loader: a torch DatasetLoader where the classes are categorical (torch.long).
    Returns:
        counts: a sorted numpy array with the class counts.

        majority_class_proportion: a float in [0., 1.] corresponding to the ratio of the most
        common class over the total number of examples.
    """
    Y = []
    for x,y in dataset_loader:

        Y.extend(list(y))

    counts = np.unique(Y, return_counts=True)[1]
    print(counts)
    majority_class_proportion = np.max(counts) / np.sum(counts)
    print("Majority class: ", majority_class_proportion)

    return counts, majority_class_proportion
