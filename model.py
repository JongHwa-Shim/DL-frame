import torch
import torch.nn as nn
import torchvision as tv

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(784, 196)
        self.act1 = nn.ELU()
        self.linear2 = nn.Linear(196, 100)
        self.act2 = nn.ELU()
        self.linear3 = nn.Linear(100, 10)
        self.LSM = nn.LogSoftmax(dim=1)


    
    def forward(self, source):
        x = self.linear1(source)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.LSM(x)
        return x

