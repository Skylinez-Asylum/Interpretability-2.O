import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GatedAutoEncoder(nn.Module):
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
        
        # Biases and additional parameters
        self.b_enc = nn.Parameter(torch.zeros(cfg['dict_dim']))
        self.b_dec = nn.Parameter(torch.zeros(cfg['activation_dim']))
        
        # Gating and magnitude parameters
        self.r_mag = nn.Parameter(torch.zeros(cfg['dict_dim']))
        self.gate_bias = nn.Parameter(torch.zeros(cfg['dict_dim']))
        self.mag_bias = nn.Parameter(torch.zeros(cfg['dict_dim']))
        
        # Normalize decoder weights
        self.W_dec.data[:] = F.normalize(self.W_dec.data, dim=-1)
        
        # Track neuron activity
        self.neuron_activity = torch.zeros(cfg['dict_dim']).to(device)
        self.step_counter = 0
        
        # Add weight decay coefficient
        self.weight_decay = cfg.get('weight_decay', 1e-5)


    def encode(self, x):
        x_enc = x @ self.W_enc + self.b_enc # a simple linear regression 
        
        pi_gate = x_enc + self.gate_bias # Gating Network
        f_gate = (pi_gate > 0).to(x_enc.dtype)
        
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias # Magnitude Network
        f_mag = F.relu(pi_mag)
        
        acts = f_gate * f_mag
        acts = acts * self.W_dec.norm(dim=1) # Anthropic team was smoking
        
        return acts

    def decode(self, f):
        
        norm = self.W_dec.norm(dim=1, keepdim=True).t() 
        f = f / norm # Anthropic team was smoking
        return f @ self.W_dec + self.b_dec


    def forward(self, x):
        x_cent = x - self.b_dec
        
        if self.training:
            x_cent = x_cent + torch.randn_like(x_cent) * 0.1
        
        acts = self.encode(x_cent)
        
        if self.training:
            acts = self.dropout(acts)
        
        x_reconstruct = self.decode(acts)
        
        l2_loss = F.mse_loss(x_reconstruct, x, reduction='mean')
        l1_loss = self.l1_coeff * acts.float().abs().sum()
        
        weight_decay_loss = self.weight_decay * (
            self.W_enc.pow(2).sum() + 
            self.W_dec.pow(2).sum() +
            self.r_mag.pow(2).sum()
        )
        
        loss = l2_loss + l1_loss + weight_decay_loss

        self.neuron_activity += (acts > 0).float().sum(0)
        self.step_counter += 1

        return loss, x_reconstruct, acts, l2_loss, l1_loss
    
    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)