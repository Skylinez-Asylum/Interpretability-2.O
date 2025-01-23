import torch
import torch.nn as nn
import torch.nn.functional as F

class JumpReluAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_dim = cfg['activation_dim']
        self.dict_dim = cfg['dict_dim']
        self.l1_coeff = cfg['l1_coeff']
        
        # Encoder parameters
        self.W_enc = nn.Parameter(torch.empty(self.activation_dim, self.dict_dim))
        self.b_enc = nn.Parameter(torch.empty(self.dict_dim))
        
        # Decoder parameters
        self.W_dec = nn.Parameter(torch.empty(self.dict_dim, self.activation_dim))
        self.b_dec = nn.Parameter(torch.empty(self.activation_dim))
        
        # Threshold for jump ReLU
        self.threshold = nn.Parameter(torch.empty(self.dict_dim))
        
        # Initialize weights
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=0.1)
        nn.init.kaiming_uniform_(self.W_dec, a=0.1)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        nn.init.uniform_(self.threshold, 0.1, 0.5)  # Initialize thresholds
        
    def encode(self, x):
        x_cent = x - self.b_dec
        pre_jump = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_jump - self.threshold)  # Jump ReLU
        return acts
    
    def decode(self, acts):
        x_reconstruct = acts @ self.W_dec + self.b_dec
        return x_reconstruct
    
    def forward(self, x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
        
        # Calculate losses
        l1_loss = self.l1_coeff * acts.abs().sum()
        l2_loss = F.mse_loss(x_reconstruct, x, reduction='mean')
        loss = l2_loss + l1_loss
        
        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    