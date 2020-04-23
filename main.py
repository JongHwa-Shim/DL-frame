from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from preprocessing import PreProcessing
from make_dataset import Mydataset
from model import Model
from save_load import save_dataset, load_dataset, save_model, load_model
from train_valid_test import train, valid
# hyperparameter
############################################################################################################################# 
DEVICE = torch.device("cuda:0")                                                                                             # 
                                                                                                                            #
LOAD_DATA = False                                                                                                           #
SAVE_DATA = True                                                                                                            #
DATASET_PATH = ""                                                                                                           #
VALID_DATASET_PATH = ""                                                                                                     #
LOAD_MODEL = False                                                                                                          #
SAVE_MODEL = True                                                                                                           #
MODEL_PATH = ""                                                                                                             #
VALID_MODEL_PATH = ""                                                                                                       #
                                                                                                                            #
TEST_MODE = False                                                                                                           #
SPLIT_DATASET = True                                                                                                        #
SPLIT_RATIO = 0.7 # 0~1                                                                                                     #
if TEST_MODE: # this is kind of trick to realize test mode, condition of test mode is 1.no training, 2.no train_dataset.    #
    SPLIT_DATASET = False                                                                                                   #
    LOAD_MODEL = True                                                                                                       #
                                                                                                                            #
                                                                                                                            #
BATCH_SIZE = None                                                                                                           #
SHUFFLE = True                                                                                                              #    
NUM_WORKERS = 4 #multithreading                                                                                             #
                                                                                                                            #
                                                                                                                            #
                                                                                                                            #
                                                                                                                            #
if LOAD_MODEL:                                                                                                              #
    model = load_model()                                                                                                    #
else:                                                                                                                       #
    model = Model().to(DEVICE)                                                                                              #
                                                                                                                            #
EPOCH = 100                                                                                                                 #
LEARNING_RATE = 0.0002                                                                                                      #
CRITERION = nn.MSELoss()                                                                                                    #
OPTIMIZER = Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)                                           #
#############################################################################################################################

# preprocessing, make or load and save dataset
############################################################################################################################# 
if LOAD_DATA == True:                                                                                                       #
                                                                                                                            #
    dataset = load_dataset(DATASET_PATH)                                                                                    #
    if SPLIT_DATASET:                                                                                                       #
        valid_dataset = load_dataset(VALID_DATASET_PATH)                                                                    #
                                                                                                                            #
else:                                                                                                                       #
                                                                                                                            #
    # preprocessing
    ############################################################# 
    source_path = None                                          #
    target_path = None                                          #
    sources, targets = PreProcessing(source_path, target_path)  #
    #############################################################

    # make dataset
    ###########################################
    transform = transforms.Compose([transforms.ToTensor()])
    if SPLIT_DATASET:
        pivot = int(len(sources) * SPLIT_RATIO)
        train_sources = sources[:pivot]
        train_targets = targets[:pivot]

        valid_sources = sources[pivot:]
        valid_targets = targets[pivot:]

        valid_dataset = Mydataset(valid_sources,valid_targets,valid_transform)      
        dataset = Mydataset(tarin_sources,train_targets,transform)
    
    else:
        dataset = MYdataset(sources,targets,transform)

    if SAVE_DATA == True:
        save_dataset(dataset, DATASET_PATH)

        if SPLIT_DATASET:
            save_dataset(valid_dataset, VALID_DATASET_PATH)
    ###########################################

dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
if SPLIT_DATASET:
    valid_dataloader = DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
#######################################################################################################################


########################################### model information
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################


########################################################## training and evaluate


epoch = range(EPOCH)
for times in epoch:

    ####################################### training
    train_losses, train_accuracy_list = train(datalaoder, model)
    #######################################

    ####################################### evaluate
    if SPLIT_DATASET:
        valid_losses, valid_accuracy_list = valid()
    #######################################

    ############################ leave log  >>>>> make to module
    train_loss = sum(train_losses)/len(train_losses)
    accuracy = train_accuracy_list.count(1)/len(train_accuracy_list)

    base = ("Epoch: {epoch:d}  Train_Loss: {loss:.8}  Train_Accuracy: {accuracy:.8}\n")
    message = base.format(epoch=times, loss=train_loss, accuracy=accuracy)
    print(message)

    if SPLIT_DATASET:
        valid_loss = sum(valid_losses)/len(valid_losses)
        valid_accuracy = valid_ accuracy_list.count(1)/len(valid_accuracy_list)
        valid_base = ({"Valid_Loss:{loss:.8} Valid_Accuracy: {accuracy:.8}"})
        valid_message = valid_base.format(loss=valid_loss, accuracy=valid_accuracy)
        print(valid_message)
    ############################
    if MODEL_SAVE == True:
        save_model(MODEL_PATH)
        if SPLIT_DATASET:
            save_model(VALID_MODEL_PATH)
    ############################ model save

    ############################
############################################################