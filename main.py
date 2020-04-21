from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.torch.device as device

from preprocessing import PreProcessing
from make_dataset import Mydataset
from model import Model


########################################### hyperparameter
DEVICE = torch.device("cuda:0")

BATCH_SIZE = None
SHUFFLE = True
NUM_WORKERS = 4 #multithreading

EPOCH = 100
LEARNING_RATE = 0.0002

criterion = nn.MSELoss()
OPTIMIZER = Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-08, weight_decay=0)
###########################################

########################################### preprocessing
source_path = None
target_path = None
sources, targets = PreProcessing(source_path, target_path)
###########################################

########################################### make dataset
transform = transforms.Compose([transforms.ToTensor()])
dataset = Mydataset(sources,targets,transform)
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
###########################################

########################################### make model
model = Model().to(DEVICE)
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################

########################################### training
model.train()

epoch = range(EPOCH)
for times in epoch:
    for data in dataloader:

        output = output.to(DEVICE)
        target = target.to(DEVICE)

        output = model(data['source'])
        target = data['target']
        loss = criterion(output, target)

        model.zero_grad()
        loss.backward()
        OPTIMIZER.step()
###########################################