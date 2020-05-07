from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from preprocessing import *
from make_dataset import *
from model import *
from save_load import *

from temp import square_plot

### hyperparameter
############################################################################################################################# 
DEVICE = torch.device("cuda:0")

LOAD_DATA = False
SAVE_DATA = False
LOAD_MODEL = False
SAVE_MODEL = False
DATASET_PATH = "./data/mnist-in-csv/mnist_test.csv"
G_PATH = "./model/generator.pkl"
D_PATH = "./model/discriminator.pkl"

TEST_MODE = False
if TEST_MODE:
    LOAD_MODEL = True


BATCH_SIZE = 64
SHUFFLE = True
NUM_WORKERS = 0 # multithreading

"""
if LOAD_MODEL:
    G = load_model(G_PATH)
    D = load_model(D_PATH)
else:
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
"""
EPOCH = 100
NUM_LEARN_D = 1
NUM_LEARN_G = 1
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0001

G_WIDTH = None
G_LENGTH = None
D_WIDTH = None
D_LENGTH = None

"""
G_OPTIMIZER = Adam(G.parameters(), lr=G_LEARNING_RATE, eps=1e-08, weight_decay=0)
D_OPTIMIZER = Adam(D.parameters(), lr=D_LEARNING_RATE, eps=1e-08, weight_decay=0)
"""
#############################################################################################################################

### preprocessing, make or load and save dataset
############################################################################################################################# 
if LOAD_DATA == True:

    dataset = load_dataset(DATASET_PATH)

else:

    ### preprocessing
    source_path = r'./data/mnist-in-csv/mnist_test.csv'
    sources, labels = PreProcessing(source_path, target_path=None, mode='csv') 

    ### make dataset
    ##############################################################################
    filter = transform_processing()
    transform = my_transform([filter.image_pixel_scale], [filter.to_LongTensor])
    # transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5],[0.5])])

    dataset = Mydataset(sources, labels, transform)

    if SAVE_DATA:
        save_dataset(dataset, DATASET_PATH)
    ###############################################################################

### make dataloader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
#######################################################################################################################


### model information
###########################################
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################

### model definition
#################################
if LOAD_MODEL:
    G = load_model(G_PATH)
    D = load_model(D_PATH)
else:
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

G_OPTIMIZER = Adam(G.parameters(), lr=G_LEARNING_RATE, eps=1e-08, weight_decay=0)
D_OPTIMIZER = Adam(D.parameters(), lr=D_LEARNING_RATE, eps=1e-08, weight_decay=0)
#################################

### training, evaluate, log and model save
##########################################################################################
epoch = range(EPOCH)
for times in epoch:
    D_losses = []
    G_losses = []
    batch_len = len(dataloader)
    for num, data in enumerate(dataloader):

        ### fitting batch size
        batch_size = data['condition'].size(0)
        REAL_ANSWER = torch.FloatTensor([[1] for _ in range(batch_size)]).to(DEVICE)
        FAKE_ANSWER = torch.FloatTensor([[0] for _ in range(batch_size)]).to(DEVICE)

        ### train discriminator
        D_input_real = [data['real'], data['condition']]
        for i in range(NUM_LEARN_D):
            D_result_real = D(D_input_real)

            G_input = data['condition']
            fake_data = G(G_input)

            D_input_fake = [fake_data, data['condition']]
            D_result_fake = D(D_input_fake)

            D_loss, D_loss_real, D_loss_fake = D_CRITERION(D_result_real, 
                                                            D_result_fake, 
                                                            real_answer=REAL_ANSWER,
                                                            fake_answer=FAKE_ANSWER
                                                            )
            D_losses.append(D_loss.data)

            D.zero_grad()
            D_loss.backward()
            D_OPTIMIZER.step()
        
        ### train generator
        for i in range(NUM_LEARN_G):
            G_input = data['condition']
            fake_data = G(G_input)

            D_input_fake = [fake_data, data['condition']]
            D_result_fake = D(D_input_fake)

            G_loss = G_CRITERION(D_result_fake, real_answer=REAL_ANSWER)
            G_losses.append(G_loss.data)

            G.zero_grad()
            G_loss.backward()
            G_OPTIMIZER.step()

    print("Epoch: " + str(times))

    ### log
    print("Average G Loss:", float(sum(G_losses)/len(G_losses)), "     Average D Loss", float(sum(D_losses)/len(D_losses)), "\n")

    ### visualization

    ### model save
    if SAVE_MODEL:
        save_model(G_PATH)
        save_model(D_PATH)
############################################################################################

### 테스트용 코드를 아예 따로 짜자...