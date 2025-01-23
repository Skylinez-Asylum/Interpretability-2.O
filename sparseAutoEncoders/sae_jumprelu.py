import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class JumpReluAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_dim = cfg['activation_dim']
        self.dict_dim = cfg['dict_dim']
        self.l1_coeff = cfg['l1_coeff']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Encoder parameters
        self.W_enc = nn.Parameter(torch.empty(self.activation_dim, self.dict_dim))
        self.b_enc = nn.Parameter(torch.empty(self.dict_dim))
        
        # Decoder parameters
        self.W_dec = nn.Parameter(torch.empty(self.dict_dim, self.activation_dim))
        self.b_dec = nn.Parameter(torch.empty(self.activation_dim))
        
        # Threshold and regularization
        self.threshold = nn.Parameter(torch.empty(self.dict_dim))
        self.dropout = nn.Dropout(p=cfg.get('dropout_rate', 0.1))
        
        # Initialization
        self.reset_parameters()
        self.neuron_activity = torch.zeros(self.dict_dim).to(self.device)
        self.step_counter = 0
        
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_enc, a=0.1)
        nn.init.kaiming_uniform_(self.W_dec, a=0.1)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)
        nn.init.uniform_(self.threshold, 0.1, 0.5)
        
    def encode(self, x):
        # Add noise during training
        if self.training:
            x = x + torch.randn_like(x) * 0.05
            
        x_cent = x - self.b_dec
        pre_jump = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_jump - self.threshold)
        
        # Track neuron activity
        if self.training:
            self.neuron_activity += (acts > 0).float().sum(0)
            self.step_counter += x.shape[0]
            
        return self.dropout(acts) if self.training else acts
    
    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec
    
    def forward(self, x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
        
        l1_loss = self.l1_coeff * acts.abs().sum()
        l2_loss = F.mse_loss(x_reconstruct, x)
        loss = l2_loss + l1_loss
        
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)