import torch
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy
import joblib
from .models import ComplexRNNModel
from .dataset import read_dataset, prepare_dataset
from .utils import plot_confusion_matrix
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

    pbar = tqdm(range(num_epochs))

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

    predicted = []
    correct = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = Variable(x_val[:, seq_start:seq_end:step_size, :])
        y_hat = model(x_val)

        current_prediction = torch.max(y_hat.data, 1)[1]
        total += y_val.size(0)
        correct += (current_prediction == y_val).sum().cpu().numpy()
        
        predicted.extend(current_prediction)

        
    accuracy = 100 * correct / float(total)
    if return_predicted:
        return accuracy, predicted
    return accuracy


def evaluate_saved_model(model_path, network_dims, test_dataset_path, training_columns, class_columns, seq_start=None,
                         seq_end=None, step_size=None, scaler_path=None, trials=None, arch=ComplexRNNModel, multiclass=False,
                         categorical_columns=(), normalisation_cols=(), save_plot_path=None, device=torch.device("cpu")):
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
        save_plot_path: If not None, the confusion matrix will be saved to this path. 
        device: (torch.device) both model and dataset will be loaded to this device.

    Returns:
        ax: the matplotlib axis from the confusion matrix plot."""

    class_columns = list(class_columns)
    training_columns = list(training_columns)

    if scaler_path:
        scaler = joblib.load(scaler_path)
        normalise_data=True
    else:
        scaler = None
        normalise_data=False

    model = arch(*network_dims)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device=device)

    if trials is None:
        trials = read_dataset(test_dataset_path)

    test_loader, _ = prepare_dataset([trials], class_columns, normalise_data=normalise_data,
                                     scaler=scaler, training_columns=training_columns, multiclass=multiclass,
                                     categorical_columns=categorical_columns, normalisation_cols=normalisation_cols,
                                     device=device)

    accuracy, predicted = evaluate(model, test_loader, return_predicted=True, seq_start=seq_start, step_size=step_size, seq_end=seq_end)

    print("Model's accuracy on test set:", accuracy)

    predicted = [pred.cpu() for pred in predicted]
    Y_test = np.concatenate([y.cpu().numpy() for x, y in test_loader])

    ax = plot_confusion_matrix(Y_test, predicted, classes=class_columns, normalize=True)
    if save_plot_path:
        plt.savefig(save_plot_path)

    return ax
