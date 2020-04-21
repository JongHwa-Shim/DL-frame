from PIL import Image
import os
import torch
from PIL import Image
from torchvision import transforms

def PreProcessing(source_path, target_path):
    ########################################### source processing
    source_list = os.listdir(source_path)
    sources = []
    for source_name in source_list:
        file_path = source_path + source_name
        source = Image.open(file_path)
        sources.append(source)
    ###########################################

    ########################################### target processing
    target_list = os.listdir(target_path)
    targets = []
    for target_name in target_list:
        file_path = target_path + target_name
        
    ###########################################

    return sources, targets