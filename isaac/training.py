import torch
from torch.autograd import Variable
from tqdm import tqdm
from copy import deepcopy

def training_loop(model, optimizer, error, train_loader, val_loader, num_epochs=200, print_stats_per_epoch=True,
                  seq_start=None, seq_end=None, step_size=None):
    """Trains a model for <num_epochs> to minimize the <error> using the <optimizer>.
    Returns a list of epoch losses (averaged over batches) as well as validation accuracy"""
    
    best_model, best_val_accuracy = None, 0
    epoch_losses = []
    epoch_accuracies = [[],[]]
    
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
        train_accuracy = evaluate(model, train_loader)
        epoch_accuracies[0].append(train_accuracy)
        val_accuracy = evaluate(model, val_loader)
        epoch_accuracies[1].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_model = deepcopy(model)
            best_val_accuracy = val_accuracy
            
        if print_stats_per_epoch:
            pbar.set_description("Train_loss (%.2f)\t Train_acc (%.2f)\t Val_acc (%.2f)" % (epoch_losses[-1], train_accuracy, val_accuracy))
    
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