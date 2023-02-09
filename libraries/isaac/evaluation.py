import torch
from torch.autograd import Variable
from copy import deepcopy
import joblib
from .models import ComplexRNNModel
from .dataset import read_dataset, prepare_dataset


def predict(model, val_loader, seq_start, seq_end, step_size, predict_seq2seq=False,
            predict_rolling_windows=False, seconds_per_window=None):

    assert not (predict_seq2seq and predict_rolling_windows)

    for i, (x_val, y_val) in enumerate(val_loader):

        x_val = Variable(x_val[:, seq_start:seq_end:step_size, :])
        if predict_seq2seq:
            y_hat = model.predict_seq2seq_in_intervals(x_val)
        elif predict_rolling_windows:
            y_hat = model.predict_seq2seq_in_rolling_windows(x_val, seconds_per_window)
        else:
            y_hat = model(x_val)

        # If y_hat is a tuple, the model is multibranch so the branch predictions will be stacked.
        if isinstance(y_hat, tuple):
            y_hat = torch.stack(y_hat, dim=2)

        if i == 0:
            all_predictions = y_hat
        else:
            all_predictions = torch.cat([all_predictions, y_hat], dim=0)

    return all_predictions


def evaluate(model, val_loader, return_predicted=False, seq_start=None, seq_end=None, step_size=None):
    """Evaluates a model given a validation set.
    Args:
        model: the trained neural network to be evaluated.
        val_loader: the validation dataset. Must be 3D (batch, sequence, features).
        return_predicted: (bool) If True, both accuracy and predictions will be returned.
        seq_start: (int) index at which the sequence to be fed starts.
        seq_end: (int) index at which the sequence to be fed ends.
        step_size: (int) step to be applied to the sequence to be fed.
    Returns:
        accuracy: (float) the percentage of accuracy achieved by the model out of 100.
        predicted: (list) if return_predicted is True, the model class predictions will be returned.
    """

    raw_predictions = predict(model, val_loader, seq_start, seq_end, step_size)
    answers = torch.max(raw_predictions, dim=1)[1]
    solutions = torch.cat([y_val for _, y_val in val_loader])
    total = len(solutions)
    correct = (answers == solutions).sum(dim=0)

    accuracy = 100 * correct.cpu().numpy() / float(total)

    if return_predicted:
        return accuracy, answers

    return accuracy


def evaluate_saved_model(model_paths, network_dims, test_dataset_path, training_columns, class_columns, seq_start=None,
                         seq_end=None, step_size=None, scaler_path=None, trials=None, arch=ComplexRNNModel, multiclass=False,
                         categorical_columns=(), normalisation_cols=(), device=torch.device("cpu"), return_test_loader=False):
    """Loads a trained model and evaluates it in a given dataset.
    Args:
        model_paths: path to the saved model or group of models sharing the same characteristics.
        network_dims: network parameters that will be passed to the network architecture.
        test_dataset_path: path to the dataset the model will be evaluated on.
        training_columns: columns in the dataset that will be used as features.
        class_columns: columns in the dataset that will be interpreted as the class.
        seq_start, seq_end, step_size: (integers) where the sequence starts and ends and the step
                                       to be applied to it.
        scaler_path: path to the saved scaler used to normalise the data. If None, the data won't
                     be normalised.
        trials: Alternatively to passing the test_dataset_path, the model can be evaluated on a set
                of already loaded trials. If both test_dataset_path and trials are not None, trials
                will be used.
        arch: model architecture.
        multiclass: argument to read_dataset. If True, indicates a multibranch network is used and
                    thus class_columns is a list of size (number_of_branches, number_of_columns_per_class)
        categorical_columns: argument to read_dataset. Indicates which columns mustn't be normalised.
        normalisation_cols: argument to read_dataset. Indicates which columns must be normalised.
        device: (torch.device) both model and dataset will be loaded to this device.
        return_test_loader: (boolean) if True, returns the test loader used to evaluate the model.

    Returns:
        accuracy: the model's accuracy.
        predicted: the model's prediction for the test dataset loaded."""

    if type(model_paths) == str:
        model_paths = [model_paths]

    class_columns = list(class_columns)
    training_columns = list(training_columns)

    if scaler_path:
        scaler = joblib.load(scaler_path)
        normalise_data=True
    else:
        scaler = None
        normalise_data=False

    if trials is None:
        trials = read_dataset(test_dataset_path)

    test_loader, _ = prepare_dataset([trials], class_columns, normalise_data=normalise_data,
                                     scaler=scaler, training_columns=training_columns, multiclass=multiclass,
                                     categorical_columns=categorical_columns, normalisation_cols=normalisation_cols,
                                     device=device)

    accuracy_list = []
    predictions_list = []

    for model_path in model_paths:
        model = arch(*network_dims)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device=device)
        model.eval()

        accuracy, predicted = evaluate(model, test_loader, return_predicted=True, seq_start=seq_start, step_size=step_size, seq_end=seq_end)

        accuracy_list.append(accuracy)
        predictions_list.append(predicted)
        # print("Model's accuracy on test set:", accuracy)

    if len(accuracy_list) == 1:
        accuracy_list = accuracy_list[0]
        predictions_list = predictions_list[0]

    if return_test_loader:
        return accuracy_list, predictions_list, test_loader

    return accuracy_list, predictions_list


def predict_with_a_group_of_saved_models(model_paths, network_dims, test_dataset_path,
                                       training_columns, class_columns, seq_start=None,
                                       seq_end=None, step_size=None, scaler_path=None,
                                       trials=None, arch=ComplexRNNModel, multiclass=False,
                                       categorical_columns=(), normalisation_cols=(),
                                       device=torch.device("cpu"), return_test_loader=False,
                                       predict_seq2seq=False, predict_rolling_windows=False,
                                       seconds_per_window=None):
    """Loads a trained model and gets their predictions for a given dataset.
    Args:
        model_paths: (list) Group of models sharing the same characteristics.
        network_dims: network parameters that will be passed to the network architecture.
        test_dataset_path: path to the dataset the model will be evaluated on.
        training_columns: columns in the dataset that will be used as features.
        class_columns: columns in the dataset that will be interpreted as the class.
        seq_start, seq_end, step_size: (integers) where the sequence starts and ends and the step
                                       to be applied to it.
        scaler_path: path to the saved scaler used to normalise the data. If None, the data won't
                     be normalised.
        trials: Alternatively to passing the test_dataset_path, the model can be evaluated on a set
                of already loaded trials. If both test_dataset_path and trials are not None, trials
                will be used.
        arch: model architecture.
        multiclass: argument to read_dataset. If True, indicates a multibranch network is used and
                    thus class_columns is a list of size (number_of_branches, number_of_columns_per_class)
        categorical_columns: argument to read_dataset. Indicates which columns mustn't be normalised.
        normalisation_cols: argument to read_dataset. Indicates which columns must be normalised.
        device: (torch.device) both model and dataset will be loaded to this device.
        return_test_loader: (boolean) if True, returns the test loader used to evaluate the model.
        predict_seq2seq: (boolean) if True, for each of the trials the prediction will be a sequence.
        predict_rolling_windows: (boolean) if True, for each of the trials the prediction will be 
                                 done in rolling windows of size <seconds_per_window>
        seconds_per_window: (int) the number of seconds in each rolling window. Only used if
                           predict_rolling_windows is True.

    Returns:
        accuracy: the model's accuracy.
        predicted: the model's prediction for the test dataset loaded."""

    assert not (predict_seq2seq and predict_rolling_windows)

    class_columns = list(class_columns)
    training_columns = list(training_columns)

    if scaler_path:
        scaler = joblib.load(scaler_path)
        normalise_data=True
    else:
        scaler = None
        normalise_data=False

    if trials is None:
        trials = read_dataset(test_dataset_path)

    test_loader, _ = prepare_dataset([trials], class_columns, normalise_data=normalise_data,
                                     scaler=scaler, training_columns=training_columns, multiclass=multiclass,
                                     categorical_columns=categorical_columns, normalisation_cols=normalisation_cols,
                                     device=device)

    predictions_list = []

    for model_path in model_paths:
        model = arch(*network_dims)
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model = model.to(device=device)
        model.eval()

        predictions = predict(model, test_loader, seq_start=seq_start, step_size=step_size,
                              seq_end=seq_end, predict_seq2seq=predict_seq2seq,
                              predict_rolling_windows=predict_rolling_windows,
                              seconds_per_window=seconds_per_window)

        predictions_list.append(predictions.detach())

    if return_test_loader:
        return predictions_list, test_loader

    return predictions_list


def get_best_model_and_its_accuracy(model_A, model_A_accuracy, model_B, model_B_accuracy):
    """Compares two models and returns a copy of the better one and its accuracy. If the models are
       equally accurate, the second model will be returned.
    Args:
        model_A: a model to be compared in terms of its accuracy.
        model_A_accuracy: the first model's accuracy.
        model_B: a model to be compared in terms of its accuracy.
        model_B_accuracy: the second model's accuracy.
    Returns:
        best_model: a copy of the model with the best accuracy.
        best_model_accuracy: the accuracy of the best model."""

    if model_A_accuracy > model_B_accuracy:
        return deepcopy(model_A), model_A_accuracy

    return deepcopy(model_B), model_B_accuracy
