from PIL import Image
import os
import torch
from PIL import Image
from torchvision import transforms
import csv


def PreProcessing(source_path, target_path=None, mode=None):
    sources = []
    targets = []

    if mode==None:
        print("please select mode")

    elif mode=="csv":
        with open(source_path) as f:
            rdr = csv.reader(f)
            next(rdr)
            for line in rdr:
                
                target = line[0]
                source = line[1:]
                target = [int(num) for num in target]
                source = [int(num) for num in source]
                targets.append(target)
                sources.append(source)
            
    elif mode=='individual':
        source_list = os.listdir(source_path)

        for source_name in source_list:
            file_path = source_path + source_name
            source = Image.open(file_path)
            sources.append(source)

        target_list = os.listdir(target_path)

        for target_name in target_list:
            file_path = target_path + target_name
            target = 3
            targets.append(target)

    elif mode=='other':
        None

    return sources, targets

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

