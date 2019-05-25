import torch
from torch.autograd import Variable
from tqdm import tqdm

def training_loop(model, optimizer, error, train_loader, val_loader, num_epochs=200):
    """Trains a model for <num_epochs> to minimize the <error> using the <optimizer>.
    Returns a list of epoch losses (averaged over batches) as well as validation accuracy"""
    
    epoch_losses = []
    epoch_accuracies = []
    
    for epoch in  tqdm(range(num_epochs)):

        epoch_loss = 0

        for x, y in train_loader:

            x = Variable(x)
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

        accuracy = evaluate(model, val_loader)

        epoch_accuracies.append(accuracy)

        print(epoch_losses[-1], epoch_accuracies[-1])
    
    return epoch_losses, epoch_accuracies


def evaluate(model, val_loader, return_predicted=False):
    predicted = []
    correct = 0
    total = 0
    for x_val, y_val in val_loader:
        x_val = Variable(x_val)
        y_hat = model(x_val)

        current_prediction = torch.max(y_hat.data, 1)[1]
        total += y_val.size(0)
        correct += (current_prediction == y_val).sum()
        
        predicted.extend(current_prediction)

        
    accuracy = 100 * correct / float(total)
    if return_predicted:
        return accuracy, predicted
    return accuracy