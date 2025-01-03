import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class SAE_Dataset(Dataset):
    def __init__(self):
        dataset = torch.rand((100, 768)) # put this in some variable
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
