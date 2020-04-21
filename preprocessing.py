from PIL import Image
import os
import torch
from PIL import Image
from torchvision import transforms

def preprocessing(source_path, target_path):
    source_path = './sample/'
    file_list = os.listdir(path)

    ########################################### source processing
    sources = []
    for file_name in file_list:
        file_path = source_path + file_name
        source = Image.open(file_path)
        sources.append(source)
    ###########################################

    ########################################### target processing
    targets = []
    ###########################################

    return sources, targets