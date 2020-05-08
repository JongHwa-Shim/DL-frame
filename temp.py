from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

def square_plot(data, path):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    if type(data) == list:
	    data = np.concatenate(data)
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an image
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imsave(path, data, cmap='gray')

#x = torch.randn(25,28,28)

im = Image.open('./sample/image_4.jpg')
trans1 = transforms.ToTensor()
im = trans1(im)
im = im.reshape((1,3,512,394))
im = [im for _ in range(25)]
im = torch.cat(im,0)
x = im
def asdf (data, path, mode='gray'):
   # batch will be n^2
   num_dim = data.ndim
   height = data.size(num_dim-2)
   width = data.size(num_dim-1)

   if height > width:
      pad = height
   else:
      pad = width
   #pad = int(pad/20) #padding ratio
   pad = 4
   n = int(np.sqrt(data.size(0)))

   if mode=='gray':
      # shape: (batch, height, width)
      #data = data.view(data.size(0), height, -1) 
      padding = ((0, 0), (pad, pad), (pad, pad))
      data = np.pad(data, padding, mode='constant', constant_values=1)

      height = data.shape[1]
      width = data.shape[2]

      sample_image = [] #generate sample image
      start = 0
      end = n
      for j in range(n):
         for i in range(height):
            row = [ element for one_image in data[start:end] for element in one_image[i]]
            sample_image.append(row)

         start = start + n
         end = end + n
            
      plt.imsave(path, sample_image, cmap="gray")

   elif mode=='RGB':
      # shape: (batch, channel=3, height, width)
      data = data.view(data.size(0), 3, height, -1) 
      padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
      data = np.pad(data, padding, mode='constant', constant_values=1)

      sample_image = [[],[],[]] #generate sample image (3, height, width)
      for k in range(3):
         start = 0
         end = n
         for j in range(n):
            for i in range(height):
               row = [ element for one_image in data[start:end][k] for element in one_image[i]]
               sample_image[k].append(row)

            start = start + n
            end = end + n

      plt.imsave(path, data, cmap="rgb")

   else:
      print("Error: this function only apply gray and RGB mode")

asdf(x,'./result/sample.jpg', mode='RGB')
#square_plot(x,'./result/sample.jpg')
