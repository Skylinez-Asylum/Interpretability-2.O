import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sae_dataset import SAE_Dataset
from torch.utils.data import Dataset, DataLoader

class TopKAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.get('k', 5)
        
        self.encoder_weight = nn.Parameter(torch.empty(
            cfg['dict_dim'], 
            cfg['activation_dim']
        ))
        
        nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        
        self.decoder_weight = nn.Parameter(self.encoder_weight.t().clone())
        self.decoder_bias = nn.Parameter(torch.zeros(cfg['activation_dim']))

    def encode(self, x):
        x_centered = x - self.decoder_bias
        pre_acts = F.linear(x_centered, self.encoder_weight)
        post_relu = F.relu(pre_acts)
        values, indices = torch.topk(post_relu, self.k, dim=-1)
        
        # Create sparse representation
        sparse_acts = torch.zeros_like(post_relu)
        sparse_acts.scatter_(-1, indices, values)
        
        return sparse_acts

    def decode(self, acts):
        x_reconstructed = F.linear(acts, self.decoder_weight) + self.decoder_bias
        return x_reconstructed

    def forward(self, x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
        
        l1_loss = self.cfg.get('l1_coeff', 3e-4) * acts.abs().sum()
        l2_loss = F.mse_loss(x_reconstruct, x, reduction='mean')
        
        weight_decay = self.cfg.get('weight_decay', 1e-5)
        weight_decay_loss = weight_decay * (
            self.encoder_weight.pow(2).sum() + 
            self.decoder_weight.pow(2).sum()
        )
        
        loss = l2_loss + l1_loss + weight_decay_loss
        return loss, x_reconstruct, acts, l2_loss, l1_loss
