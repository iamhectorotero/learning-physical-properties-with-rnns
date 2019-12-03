import numpy as np
from scipy.stats import ttest_ind

def z_test(correct_answers, answers_first_model, answers_second_model):
    """Compares two models given their answers to a dataset and the true answers.
    The null hypothesis is that both models are equally accurate. In the alternative hypothesis,
    the second model is better than the first one. As a rule of thumb, if the value returned
    is smaller than -1.645, the difference is significant (i.e. the second model is better).
    Otherwise, no significant difference is found between models.

    Args:
        correct_answers: the dataset's correct answers.
        answers_first_model: the answers of the first model to the dataset.
        answers_second_model: the answers of the second model to the dataset.

    Returns:
        Z: the statistic calculated on the test. """

    n = len(correct_answers)

    x1 = (answers_first_model == correct_answers).sum()
    x2 = (answers_second_model == correct_answers).sum()

    p1 = x1 / float(n)
    p2 = x2 / float(n)

    p = (x1 + x2) /(2*n)
    Z = (p1 - p2) / np.sqrt(2*p*(1-p) / n)

    return Z


def is_best_model_significantly_better(accuracies):
    """Compares two accuracy lists and checks whether the model with the largest accuracy is
    significantly better than the rest.

    Args:
        model_accuracies: A list of models' accuracies to compare. Every row must also include a
        label, e.g.: [("GRU", [0.25, 0.5, 0.4]), ("RNN", [0.33, 0.4, 0.25])]"""

    best_mean_accuracy = 0.
    best_label = ""
    best_accuracy_list = []

    for label, accuracy_list in accuracies:
        if np.mean(accuracy_list) > best_mean_accuracy:
            best_mean_accuracy = np.mean(accuracy_list)
            best_accuracy_list = accuracy_list
            best_label = label
    significantly_best = True

    for label, accuracy_list in accuracies:
        if label != best_label:
            statistic, pvalue = ttest_ind(best_accuracy_list, accuracy_list)

            if statistic < 0 or pvalue > 0.05:
                print(best_label+" not significantly better than "+label, pvalue)
                significantly_best = False

    if significantly_best:
        print(best_label + " is significantly better than the rest")
