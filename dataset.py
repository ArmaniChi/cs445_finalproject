from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import torch.utils.data as data

class ColourTransferData(Dataset):
    def __init__(self, data_dir, transform=None, loader=Image.open.convert('RGB')):
        
        self.transform = transform
        self.loader = loader

        raw, a, b, c, d, e = get_data(data_dir)
    
    def __getitem__(self, index):
        path = self.data[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.data)
    
    def get_data(self):
        return self.data
    

def get_data(data_dir):
    """
    Get the data from the data directory
    data is stored in 5 subfolders: raw, a, b, c, d, e
    indicating the source (raw) and target (a, b, c, d, e) images
    target images are the same as the source image but retouched by different people
    """
    images = {
        'raw': [],
        'a': [],
        'b': [],
        'c': [],
        'd': [],
        'e': []
    }

    # for each subfolder
    for folder in os.listdir(data_dir):
        i = 0

        # for each image in the subfolder
        for file in os.listdir(os.path.join(data_dir, folder)):

            if folder in images:
                # read the image as a numpy array
                img = np.array(Image.open(os.path.join(data_dir, folder, file)))
                # add the image to the list
                images[folder].append(img)

            else:
                raise ValueError('Unknown folder: {}'.format(folder))
            
            if i % 100 == 0:
                print('Read {} images from folder {}'.format(i, folder))

            i += 1

    return images['raw'], images['a'], images['b'], images['c'], images['d'], images['e']