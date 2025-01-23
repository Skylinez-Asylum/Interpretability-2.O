import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ReluAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.l1_coeff = cfg['l1_coeff']
        self.dropout = nn.Dropout(p=cfg.get('dropout_rate', 0.1))
        
        # Initialize weights with smaller values
        init_scale = 0.1
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg['activation_dim'], cfg['dict_dim'])) * init_scale)
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(
            torch.empty(cfg['dict_dim'], cfg['activation_dim'])) * init_scale)
        
        self.b_enc = nn.Parameter(torch.zeros(cfg['dict_dim']))
        self.b_dec = nn.Parameter(torch.zeros(cfg['activation_dim']))
        
        # Normalize decoder weights
        self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=-1)
        
        self.neuron_activity = torch.zeros(cfg['dict_dim']).to(device)
        self.step_counter = 0
        
        # Add weight decay coefficient
        self.weight_decay = cfg.get('weight_decay', 1e-5)

    def forward(self, x):
        x_cent = x - self.b_dec
        
        # Add noise during training for better generalization
        if self.training:
            x_cent = x_cent + torch.randn_like(x_cent) * 0.1
            
        pre_acts = x_cent @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        
        # Apply dropout during training
        if self.training:
            acts = self.dropout(acts)
            
        
        x_reconstruct = acts @ self.W_dec + self.b_dec
        
        # Calculate losses
        l1_loss = self.l1_coeff * acts.float().abs().sum()
        l2_loss = F.mse_loss(x_reconstruct, x, reduction='mean')
        
        # Add L2 regularization
        weight_decay_loss = self.weight_decay * (
            self.W_enc.pow(2).sum() + 
            self.W_dec.pow(2).sum()
        )
        
        loss = l2_loss + l1_loss + weight_decay_loss

        # Update neuron activity
        self.neuron_activity += (acts > 0).float().sum(0)
        self.step_counter += 1

        return loss, x_reconstruct, acts, l2_loss, l1_loss


    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)