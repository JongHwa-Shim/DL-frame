import numpy as np
def visualization (data, path, mode='gray'):
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

        height = pad + height + pad
        width = pad + width + pad

        data = data.veiw(3, n*height, n*width)
        plt.imsave(path, data, cmap="RGB")

    else:
        print("Error: this function only apply gray and RGB mode")



