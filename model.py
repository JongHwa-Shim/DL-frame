import torch
import torch.nn as nn
import torchvision as tv
from preprocessing import *

def G_input_processing(model, condition):
    if condition:
        batch_size = condition.size(0)
        z = torch.randn((batch_size,100)).cuda()

        g_in = torch.cat((model.label_emb(condition), z), -1)

    else:
        None
    
    return g_in


def D_input_processing(model, data, condition):
    #D_input = torch.cat((real_data, condition), 1)
    if condition:
        batch_size = condition.size(0)
        d_in = torch.cat((model.label_emb(condition), data), -1)

    else:
        None
    
    return d_in


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10,10)
        self.model = nn.Sequential(
            nn.Linear(110,128),
            nn.BatchNorm1d(128,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(128,256),
            nn.BatchNorm1d(256,0.8),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,512),
            nn.BatchNorm1d(512,0.8),
            nn.LeakyReLU(0.2,inplace=True),   
            nn.Linear(512,1024),
            nn.BatchNorm1d(1024,0.8),
            nn.LeakyReLU(0.2,inplace=True),   
            nn.Lienar(1024,794),
            nn.Tanh()
        )
    
    def forward(self, G_input):
        g_in = G_input_processing(self, G_input)
        g_out = self.model(g_in)
        return g_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.condition_embedding = nn.Embedding(10,10)
        self.model = nn.Sequential(
            nn.Linear(794,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,1)
        )
    
    def forward(self, D_input):
        d_in = D_input_processing(self, D_input)
        d_out = self.model(d_in)
        return d_out


def G_CRITERION(D_result_fake, real_answer=None):
    loss_function = nn.MSELoss()
    G_loss = loss_function(D_result_fake, real_answer)
    return G_loss


def D_CRITERION(D_result_real, D_result_fake, real_answer=None, fake_answer=None):
    loss_function = nn.MSELoss()
    D_loss_real = loss_function(D_result_real, real_answer)
    D_loss_fake = loss_function(D_result_fake, fake_answer)
    D_loss = D_loss_real + D_loss_fake

    return D_loss, D_loss_real, D_loss_fake





    
        
