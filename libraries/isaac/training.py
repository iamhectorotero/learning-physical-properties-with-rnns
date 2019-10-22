import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy

from .constants import TQDM_DISABLE
from .evaluation import evaluate

def training_loop(model, optimizer, error, train_loader, val_loader, num_epochs=200, print_stats_per_epoch=True,
                  seq_start=None, seq_end=None, step_size=None, patience=np.inf, multibranch=False):
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
        multibranch: whether the trained model is a multibranch neural network or not.
    Returns:
        epoch_losses: the model's losses on the training dataset after every epoch.
        epoch_accuracies: (2D list) the model's accuracy on the training and validation dataset.
        best_model: the model with the best validation accuracy.
    """

    epoch_losses = []
    epoch_accuracies = [[],[]]
    best_model = [None, None] if multibranch else [None]
    best_val_accuracy = [0., 0.] if multibranch else [0.]
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

            if not multibranch:
                y_hat = (y_hat,)
                y = y[:, None]

            # Calculate softmax and cross entropy loss for each branch
            batch_loss = []
            for i, branch_y_hat in enumerate(y_hat):
                batch_loss.append(error(branch_y_hat, y[:, i]))

            torch.autograd.backward(batch_loss)
            # Update parameters
            optimizer.step()

            epoch_loss += np.array([loss.item() for loss in batch_loss])

        normalised_batch_loss = epoch_loss / len(train_loader)
        epoch_losses.append(normalised_batch_loss)

        model.eval()
        train_accuracy = evaluate(model, train_loader, seq_start=seq_start, seq_end=seq_end, step_size=step_size)
        epoch_accuracies[0].append(train_accuracy)
        val_accuracy = evaluate(model, val_loader, seq_start=seq_start, seq_end=seq_end, step_size=step_size)
        epoch_accuracies[1].append(val_accuracy)

        if not multibranch:
            val_accuracy = [val_accuracy]

        improvement_in_this_epoch = False
        for i, (branch_accuracy, branch_best_accuracy) in enumerate(zip(val_accuracy, best_val_accuracy)):
            if branch_accuracy > branch_best_accuracy:
                best_val_accuracy[i] = branch_accuracy
                best_model[i] = deepcopy(model)
                epochs_without_improvement = 0
                improvement_in_this_epoch = True

        if not improvement_in_this_epoch:
            epochs_without_improvement += 1
            if epochs_without_improvement > patience:
                break

        if print_stats_per_epoch:
            description = "Train_loss: (%s) Train_acc: (%s) Val_acc: (%s)" % (normalised_batch_loss, train_accuracy, val_accuracy)
            pbar.set_description(description)

    if not multibranch:
        best_model = best_model[0]

    epoch_losses = np.array(epoch_losses)
    epoch_accuracies[0] = np.array(epoch_accuracies[0])
    epoch_accuracies[1] = np.array(epoch_accuracies[1])

    return epoch_losses, epoch_accuracies, best_model
