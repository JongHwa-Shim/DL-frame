from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from preprocessing import PreProcessing
from make_dataset import Mydataset, trans
from model import Model
from save_load import save_dataset, load_dataset, save_model, load_model
from train_valid_test import train, valid
from message import leave_log
# hyperparameter
############################################################################################################################# 
DEVICE = torch.device("cuda:0")

LOAD_DATA = False
SAVE_DATA = False
DATASET_PATH = ""
VALID_DATASET_PATH = ""
LOAD_MODEL = False
SAVE_MODEL = True
MODEL_PATH = ""

TEST_MODE = False
SPLIT_DATASET = True
SPLIT_RATIO = 0.7 # 0~1
if TEST_MODE: # this is kind of trick to realize test mode, condition of test mode is 1.no training, 2.no train_dataset.
    SPLIT_DATASET = False
    LOAD_MODEL = True


BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 0 # multithreading

if LOAD_MODEL:
    model = load_model()
else:
    model = Model().to(DEVICE)

EPOCH = 100
LEARNING_RATE = 0.0002
CRITERION = nn.NLLLoss()
OPTIMIZER = Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)
#############################################################################################################################

# preprocessing, make or load and save dataset
############################################################################################################################# 
if LOAD_DATA == True:

    dataset = load_dataset(DATASET_PATH)
    if SPLIT_DATASET:
        valid_dataset = load_dataset(VALID_DATASET_PATH)

else:

    # preprocessing
    source_path = r'./data/mnist-in-csv/mnist_test.csv'
    target_path = None
    sources, targets = PreProcessing(source_path, target_path,mode='csv') 

    # make dataset
    ##############################################################################

    #transform = transforms.Compose([transforms.ToTensor()])
    transform = trans()

    if SPLIT_DATASET:
        pivot = int(len(sources) * SPLIT_RATIO)
        train_sources = sources[:pivot]
        train_targets = targets[:pivot]

        valid_sources = sources[pivot:]
        valid_targets = targets[pivot:]

        valid_dataset = Mydataset(valid_sources,valid_targets,transform)      
        dataset = Mydataset(train_sources,train_targets,transform)
    
    else:
        dataset = MYdataset(sources,targets,transform)

    if SAVE_DATA == True:
        save_dataset(dataset, DATASET_PATH)

        if SPLIT_DATASET:
            save_dataset(valid_dataset, VALID_DATASET_PATH)
    ###############################################################################

# make dataloader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
if SPLIT_DATASET:
    valid_dataloader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
#######################################################################################################################


# model information
###########################################
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################

# training, evaluate, log and model save
##########################################################################################
epoch = range(EPOCH)
if SPLIT_DATASET:
    best_valid_loss = float('inf')
else: 
    best_train_loss = float('inf')

for times in epoch:

    # training
    train_losses, train_accuracy_list = train(dataloader, model, CRITERION, OPTIMIZER, DEVICE)

    # evaluate
    if SPLIT_DATASET:
        valid_losses, valid_accuracy_list = valid(valid_dataloader, model, CRITERION, OPTIMIZER, DEVICE)

    # leave log
    train_message, train_loss, train_accuracy = leave_log(train_losses, train_accuracy_list, epoch)
    print(train_message)

    if SPLIT_DATASET:
        valid_message, valid_loss, valid_accuracy = leave_log(valid_losses, valid_accuracy_list)
        print('\n' + valid_message)

    # model save
    if MODEL_SAVE == True:
        if SPLIT_DATASET:
            if best_valid_loss > valid_loss:
                save_model(MODEL_PATH)
                best_valid_loss = valid_loss
        else:
            if best_train_loss > train_loss:
                save_model(MODEL_PATH)
                best_train_loss = train_loss
############################################################################################