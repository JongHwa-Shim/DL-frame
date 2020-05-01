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
        d_in = torch.cat((D_input['real'],self.condition_embedding(D_input['condition'])),1)
        d_out = self.model(d_in)
        return x


class g_criterion(object):
    def __init__(self,loss_function=None):
        self.loss_function = loss_function
    
    def calc_loss(self, D_result_fake, real_answer):
        #calculate with your own way
        G_loss = self.loss_function(D_result_fake, real_answer)
        return G_loss

class d_criterion(object):
    def __init__(self, loss_function=None):
        self.loss_function = loss_function


    def calc_loss(self, D_result_real, D_result_fake, real_answer, fake_answer):
        #calculate with your own way
        D_loss_real = self.loss_fucntion(D_result_real, real_answer)
        D_loss_fake = self.loss_function(D_result_fake, fake_answer)
        D_loss = D_loss_real + D_loss_fake
        return D_loss, D_loss_real, D_loss_fake

    
        
