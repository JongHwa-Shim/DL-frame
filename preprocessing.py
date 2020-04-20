from __future__ import print_function, division
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

# 경고 메시지 무시하기
import warnings
warnings.filterwarnings("ignore")



root_dir = './sample/'
im = io.imread(root_dir+'image_2.jpg')

im2 = Image.new("RGB", (256,256),(102,153,255))
pix = im2.load()


a=1