from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
import torch.utils.data as data

class ColourTransferData(Dataset):
    def __init__(self, data, transform=None, loader=Image.open.convert('RGB')):
        
        self.transform = transform
        self.loader = loader
        self.data = np.load(data)['arr_0']
    
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