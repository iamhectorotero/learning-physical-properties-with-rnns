import numpy as np

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
