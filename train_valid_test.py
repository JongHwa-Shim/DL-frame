import torch
import torch.nn as nn
from torch.optim import Adam

def train(dataloader, model, CRITERION, OPTIMIZER, DEVICE):
    train_losses = []
    train_accuracy_list = []

    model.train()
    for data in dataloader: # check what returned by dataloader is

        input = data['source'].to(DEVICE)
        target = data['target'].to(DEVICE)

        output = model(input)

        target = torch.squeeze(target, dim=1) # particular processing (not essential)
        loss = CRITERION(output, target)

        model.zero_grad()
        loss.backward()
        OPTIMIZER.step()

        train_losses.append(loss.data.item())

        batch_size = output.shape[0]
        max_index = torch.max(output,1).indices

        for i in range(batch_size): #if index of max value of output equal target, treat it as correct
            if max_index[i] == target[i]:
                train_accuracy_list.append(1)
            else:
                train_accuracy_list.append(0)
    
    return train_losses, train_accuracy_list

def valid(valid_dataloader, model, CRITERION, OPTIMIZER, DEVICE):
    valid_losses = []
    valid_accuracy_list = []

    model.eval()
    for data in valid_dataloader:
        input = data['source'].to(DEVICE)
        target = data['target'].to(DEVICE)

        output = model(input)

        target = torch.squeeze(target, dim=1) # particular processing (not essential)
        loss = CRITERION(output, target)

        valid_losses.append(loss.data.item())

        batch_size = output.shape[0]
        max_index = torch.max(output,1).indices

        for i in range(batch_size): #if index of max value of output equal target, treat it as correct
            if max_index[i] == target[i]:
                valid_accuracy_list.append(1)
            else:
                valid_accuracy_list.append(0)
    
    return valid_losses, valid_accuracy_list