import torch
import torch.nn as nn
import torchvision as tv

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(784, 196)
        self.act1 = nn.ELU()
        self.linear2 = nn.Linear(196, 100)
        self.act2 = nn.ELU()
        self.linear3 = nn.Linear(100, 10)
        self.LSM = nn.LogSoftmax(dim=1)


    
    def forward(self, G_input):
        x = self.linear1(G_input)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.LSM(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        None
    
    def forward(self, D_input):
        None

class Loss_func(object):
    def __init__(self,other_loss=None):
        self.LF = other_loss
    
    def Loss_calc(*args):
        if self.LF is not None:
            loss = self.LF(*args)
            return loss
        else:
            None
