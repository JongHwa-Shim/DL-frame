from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from preprocessing import PreProcessing
from make_dataset import Mydataset, self_transform
from model import *
from save_load import save_dataset, load_dataset, save_model, load_model
from train_valid_test import train, valid
from message import leave_log
# hyperparameter
############################################################################################################################# 
DEVICE = torch.device("cuda:0")

LOAD_DATA = False
SAVE_DATA = False
LOAD_MODEL = False
SAVE_MODEL = False
DATASET_PATH = ""
G_PATH = ""
D_PATH = ""

TEST_MODE = False
if TEST_MODE:
    LOAD_MODEL = True


BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 0 # multithreading

if LOAD_MODEL:
    G = load_model(G_PATH)
    D = load_model(D_PATH)
else:
    G = Genrator().to(DEVICE)
    D = Discriminator().to(DEVICE)

EPOCH = 100
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0001

G_CRITERION = Loss_func()
D_CRITERION = Loss_func(nn.NLLLoss())

G_WIDTH = None
G_LENGTH = None
D_WIDTH = None
D_LENGTH = None

G_OPTIMIZER = Adam(G.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)
D_OPTIMIZER = Adam(D.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)
#############################################################################################################################

# preprocessing, make or load and save dataset
############################################################################################################################# 
if LOAD_DATA == True:

    dataset = load_dataset(DATASET_PATH)

else:

    # preprocessing
    source_path = r'./data/mnist-in-csv/mnist_test.csv'
    sources = PreProcessing(source_path) 

    # make dataset
    ##############################################################################

    transform = transforms.Compose([self_transform()])

    dataset = MYdataset(sources,transform)

    if SAVE_DATA == True:
        save_dataset(dataset, DATASET_PATH)
    ###############################################################################

# make dataloader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
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
for times in epoch:
    D_losses = []
    G_losses = []
    for data in dataloader:
        ### train discriminator
        D_input_real = D_input_processing(data['real'],data['condition'])
        for i in range(NUM_LEARN_D):
            D_result_real = D(D_input_real)
            #D_loss_real = CRITERION(D_result_real,mode='real')

            G_input = G_input_processing(data['condition'], width=G_WIDTH, length=G_LENGTH)
            fake_data = G(G_input)

            D_input_fake = D_input_processing(fake_data, data['condition'], width=D_WIDTH, length=D_LENGTH)
            D_result_fake = D(D_input_fake)
            #D_loss_fake = CRITERION(D_result_fake,mode='fake')

            D_loss, D_loss_real, D_loss_fake = D_CRITERION(D_result_real, D_result_fake)
            D_losses.append(D_loss.data)

            D.zero_grad()
            D_loss.backward()
            D_OPTIMIZER.step()
        
        ### train generator
        for i in range(NUM_LEARN_G):
            G_input = G_input_processing(data['condition'])
            fake_data = G(G_input)

            D_result_fake = D(fake_data)
            G_loss = G_CRITERION(D_result_fake)
            G_losses.append(G_loss.data)

            G.zero_grad()
            G_loss.backward()
            G_OPTIMIZER.step()

    ### log
    print("G Loss:", sum(G_losses)/len(G_losses), "     D Loss", sum(D_losses)/len(D_losses), "\n")

    ### visualization

    # model save
    if SAVE_MODEL == True:
        save_model(G_PATH)
        save_model(D_PATH)
############################################################################################