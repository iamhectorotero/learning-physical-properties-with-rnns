import torch
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy
import joblib
from .models import ComplexRNNModel
from .dataset import read_dataset, prepare_dataset
from .constants import TQDM_DISABLE
import numpy as np
import matplotlib.pyplot as plt


def training_loop(model, optimizer, error, train_loader, val_loader, num_epochs=200, print_stats_per_epoch=True,
                  seq_start=None, seq_end=None, step_size=None, patience=np.inf):
    """Trains a model to minimize the error on the training set during the specified set of epochs.
    The model is evaluated on the validation set after every epoch and the model with the best
    validation accuracy is returned.
    Args:
        model: the neural network to be fitted.
        optimizer: the training optimizer (e.g. Adam).
        error: the error to be minimised when fitting the network.
        train_loader: the training dataset (a 3D PyTorch DatasetLoader).
        val_loader: the validation dataset (a 3D PyTorch DatasetLoader).
        num_epochs: the number of epochs the model will be trained.
        print_stats_per_epoch: whether to print accuracy and loss statistics after every epoch.
        seq_start, seq_end, step_size: the start, end and step applied to each example's sequence.
        patience: the number of epochs without validation accuracy improvement after which the
        training will stop. Defaults to infinity.
    Returns:
        epoch_losses: the model's losses on the training dataset after every epoch.
        epoch_accuracies: (2D list) the model's accuracy on the training and validation dataset.
        best_model: the model with the best validation accuracy.
    """

    best_model, best_val_accuracy = None, 0
    epoch_losses = []
    epoch_accuracies = [[],[]]
    epochs_without_improvement = 0

    pbar = tqdm(range(num_epochs), disable=TQDM_DISABLE)

    for epoch in pbar:

        model.train()

        epoch_loss = 0

        for x, y in train_loader:

            x = Variable(x[:, seq_start:seq_end:step_size, :])
            y = Variable(y)
            # Clear gradients
            optimizer.zero_grad()
            # Forward propagation
            y_hat = model(x)
            # Calculate softmax and cross entropy loss
            loss = error(y_hat, y)
            # Calculating gradients
            loss.backward()
            # Update parameters
            optimizer.step()

            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_loader))

        model.eval()
        train_accuracy = evaluate(model, train_loader, seq_start=seq_start, seq_end=seq_end, step_size=step_size)
        epoch_accuracies[0].append(train_accuracy)
        val_accuracy = evaluate(model, val_loader, seq_start=seq_start, seq_end=seq_end, step_size=step_size)
        epoch_accuracies[1].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_model = deepcopy(model)
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break

        if print_stats_per_epoch:
            this_epoch_loss = epoch_losses[-1]
            pbar.set_description("Train_loss (%.2f)\t Train_acc (%.2f)\t Val_acc (%.2f)" % (this_epoch_loss, train_accuracy, val_accuracy))

    return epoch_losses, epoch_accuracies, best_model


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

    total = 0

    for i, (x_val, y_val) in enumerate(val_loader):

        x_val = Variable(x_val[:, seq_start:seq_end:step_size, :])
        y_hat = model(x_val)

        # If y_hat is a tuple, the model is multibranch so the branch predictions will be stacked.
        if isinstance(y_hat, tuple):
            y_hat = torch.stack(y_hat, dim=2)

        current_predictions = torch.max(y_hat.data, dim=1)[1]
        correct = (current_predictions == y_val).sum(dim=0)

        if i == 0:
            all_predictions = current_predictions
            all_correct = correct
        else:
            all_predictions = torch.cat([all_predictions, current_predictions], dim=0)
            all_correct += correct

        total += y_val.size(0)


    accuracy = 100 * all_correct.cpu().numpy() / float(total)

    if return_predicted:
        return accuracy, all_predictions

    return accuracy


def evaluate_saved_model(model_path, network_dims, test_dataset_path, training_columns, class_columns, seq_start=None,
                         seq_end=None, step_size=None, scaler_path=None, trials=None, arch=ComplexRNNModel, multiclass=False,
                         categorical_columns=(), normalisation_cols=(), device=torch.device("cpu"), return_test_loader=False):
    """Loads a trained model and evaluates it in a given dataset.
    Args:
        model_path: path to the saved model.
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

    class_columns = list(class_columns)
    training_columns = list(training_columns)

    if scaler_path:
        scaler = joblib.load(scaler_path)
        normalise_data=True
    else:
        scaler = None
        normalise_data=False

    model = arch(*network_dims)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if trials is None:
        trials = read_dataset(test_dataset_path)

    test_loader, _ = prepare_dataset([trials], class_columns, normalise_data=normalise_data,
                                     scaler=scaler, training_columns=training_columns, multiclass=multiclass,
                                     categorical_columns=categorical_columns, normalisation_cols=normalisation_cols,
                                     device=device)

    accuracy, predicted = evaluate(model, test_loader, return_predicted=True, seq_start=seq_start, step_size=step_size, seq_end=seq_end)
    print("Model's accuracy on test set:", accuracy)

    if return_test_loader:
        return accuracy, predicted, test_loader

    return accuracy, predicted


