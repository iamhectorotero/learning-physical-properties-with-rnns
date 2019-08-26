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
    """Trains a model for <num_epochs> to minimize the <error> using the <optimizer>.
    Returns a list of epoch losses (averaged over batches) as well as validation accuracy"""

    best_model, best_val_accuracy = None, 0
    epoch_losses = []
    epoch_accuracies = [[],[]]
    epochs_without_improvement = 0

    pbar = tqdm(range(num_epochs))

    for epoch in pbar:
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
            if len(epoch_losses) == 0:
                this_epoch_loss = -1.
            else:
                this_epoch_loss = epoch_losses[-1]
            pbar.set_description("Train_loss (%.2f)\t Train_acc (%.2f)\t Val_acc (%.2f)" % (this_epoch_loss, train_accuracy, val_accuracy))

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
        
    return epoch_losses, epoch_accuracies, best_model


def evaluate(model, val_loader, return_predicted=False, seq_start=None, seq_end=None, step_size=None):
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
                         categorical_columns=(), normalisation_cols=(), save_plot_path=None):
    
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
    model = model.cuda()
    
    if trials is None:
        trials = read_dataset(test_dataset_path)
        
    test_loader, _ = prepare_dataset([trials], class_columns, normalise_data=normalise_data, 
                                     scaler=scaler, training_columns=training_columns, multiclass=multiclass,
                                     categorical_columns=categorical_columns, normalisation_cols=normalisation_cols)

    
    accuracy, predicted = evaluate(model, test_loader, return_predicted=True, seq_start=seq_start, step_size=step_size, seq_end=seq_end)
    
    print("Model's accuracy on test set:", accuracy)
    
    predicted = [pred.cpu() for pred in predicted]
    Y_test = np.concatenate([y.cpu().numpy() for x, y in test_loader])
    
    # plot_confusion_matrix(Y_test, predicted, classes=class_columns, normalize=False)
    # if len(save_plot_paths) > 0:
    #    matplotlib2tikz.save(save_plot_paths[0])
    ax = plot_confusion_matrix(Y_test, predicted, classes=class_columns, normalize=True)
    if save_plot_path:
        plt.savefig(save_plot_path)
        
    return ax
