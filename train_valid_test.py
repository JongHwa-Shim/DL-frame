import torch.nn as nn
from torch.optim import Adam

def train(dataloader, model, DEVICE):
    train_losses = []
    train_accuracy_list = []

    model.train()
    for data in dataloader: # check what returned by dataloader is

        input = data['source'].to(DEVICE)
        target = data['target'].to(DEVICE)

        output = model(input)
        loss = CRITERION(output, target) ################### CRITERION has not defined, 이거 main에서 가져오든지 어떻게 처리해줘야 함

        model.zero_grad()
        loss.backward()
        OPTIMIZER.step()

        train_losses.append(loss.data)
        if torch.max(output,1)[0] == target:    #if index of max value of output equal target, treat it as correct
            train_accuracy_list.append(1)             
        else:
            train_accuracy_list.append(0)
    
    return train_losses, train_accuracy_list

def valid(valid_dataloader, model):
    valid_losses = []
    valid_accuracy_list = []

    model.eval()
    for data in valid_dataloader:
        input = data['source'].to(DEVICE)
        target = data['target'].to(DEVICE)

        output = model(input)
        loss = CRITERION(output, target)

        valid_losses.append(loss.data)
        if torch.max(output,1)[0] == target:
            valid_accuracy_list.append(1)
        else:
            valid_accuracy_list.append(0)
    
    return valid_losses, valid_accuracy_list