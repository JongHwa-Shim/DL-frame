from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch

from preprocessing import PreProcessing
from make_dataset import Mydataset
from model import Model
from save_load import save_dataset, load_dataset, save_model, load_model


########################################### hyperparameter
DEVICE = torch.device("cuda:0")
LOAD_DATA = False
SAVE_DATA = True
SAVED_DATASET_PATH = ""

BATCH_SIZE = None
SHUFFLE = True
NUM_WORKERS = 4 #multithreading

EPOCH = 100
LEARNING_RATE = 0.0002

model = Model().to(DEVICE)
criterion = nn.MSELoss()
OPTIMIZER = Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)
###########################################


######################################################### preprocessing, make or load and save dataset
if LOAD_DATA == True:

    dataset = load_dataset(SAVED_DATASET_PATH)

else:

    ########################################### preprocessing
    source_path = None
    target_path = None
    sources, targets = PreProcessing(source_path, target_path)
    ###########################################

    ########################################### make dataset
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = Mydataset(sources,targets,transform)

    if SAVE_DATA == True:
        save_dataset(dataset, SAVED_DATASET_PATH)
    ###########################################

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
#########################################################


########################################### model information
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################


########################################################## training
model.train()


epoch = range(EPOCH)
for times in epoch:

    train_losses = []
    accuracy_list = []

    for data in dataloader: # check what returned by dataloader is

        output = output.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data['source'])
        target = data['target']
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()
        OPTIMIZER.step()

        train_losses.append(loss.data)
        if torch.max(output,1)[0] == target: #if index of max value of output equal target, treat it as correct
            accuracy_list.append(1)
        else:
            accuracy_list.append(0)
        
    ############################ leave log  
    train_loss = sum(train_losses)/len(train_losses)
    accuracy = accuracy_list.count(1)/len(accuracy_list)

    base = ("Epoch: {epoch:d}  Loss: {loss:.8}  Accuracy: {accuracy:.8}")
    message = base.format(epoch=times, loss=train_loss, accuracy=accuracy)
    
    print(message)
    ############################

    ############################ model save
    
    ############################
############################################################