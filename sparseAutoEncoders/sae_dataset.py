import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SAE_Dataset(Dataset):
    def __init__(self, path, len=0):
        if len == 0: dataset = np.load(path)
        else: 
            dataset = np.load(path)[:len]
            
        
        self.x = dataset
        self.y = dataset
        self.m = dataset.shape[0]
        print('Dataset:', dataset.shape)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.m
    


class SAE_RandomDataset(Dataset):
    def __init__(self, len=100):
        dataset = torch.rand((len, 768)) # put this in some variable
        
        self.x = dataset
        self.y = dataset
        self.m = dataset.shape[0]
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.m


if __name__ == '__main__':
    ds = SAE_Dataset()
    print(ds.x)
    print(ds.y)
