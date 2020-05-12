import torch
import torch.nn as nn
import torchvision as tv

def G_input_processing(model, device, condition, latent=None, mode='train'):
    condition = condition.to(device)

    if condition is not None:
        batch_size = condition.size(0)
        z = torch.randn((batch_size,100)).to(device)

        if mode=='val':
            z = latent

        z = model.zLinear(z)
        G_input = model.relu(torch.cat((model.condition_embed(condition).view(batch_size,-1), z), -1))

    else:
        None
    
    return G_input


def D_input_processing(model, device, data, condition):
    #D_input = torch.cat((real_data, condition), 1)
    data = data.to(device)
    condition = condition.to(device)

    if condition is not None:
        batch_size = condition.size(0)
        D_input = torch.cat((model.condition_embed(condition).view(batch_size,-1), data), -1)

    else:
        None
    
    return D_input


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.condition_embed = nn.Embedding(10,2000)
        self.zLinear = nn.Linear(100,200)
        self.relu = nn.LeakyReLU(0.2,inplace=True)
        self.model = nn.Sequential(
            nn.Linear(2200,1500),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1500,1000),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1000,800),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(800,784),
            nn.Tanh()
        )
    
    def forward(self, G_input):
        G_out = self.model(G_input)
        return G_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.condition_embed = nn.Embedding(10,200)

        self.model = nn.Sequential(
            nn.Linear(984,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
    
    def forward(self, D_input):
        D_out = self.model(D_input)
        return D_out


def G_CRITERION(D_result_fake, real_answer=None):
    loss_function = nn.BCELoss()
    G_loss = loss_function(D_result_fake, real_answer)
    return G_loss


def D_CRITERION(D_result_real, D_result_fake, real_answer=None, fake_answer=None):
    loss_function = nn.BCELoss()
    D_loss_real = loss_function(D_result_real, real_answer)
    D_loss_fake = loss_function(D_result_fake, fake_answer)
    D_loss = D_loss_real + D_loss_fake

    return D_loss, D_loss_real, D_loss_fake





    
        
