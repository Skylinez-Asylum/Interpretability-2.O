import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class GatedAutoEncoder(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        self.activation_dim = cfg['activation_dim']
        self.dict_dim = cfg['dict_dim']
        self.l1_coeff = cfg['l1_coeff']

        self.encoder = nn.Linear(self.activation_dim,self.dict_dim,bias=False)
        self.r_mag = nn.Parameter(torch.empty(self.dict_dim))
        self.gate_bias = nn.Parameter(torch.empty(self.dict_dim))
        self.mag_bias = nn.Parameter(torch.empty(self.dict_dim))
        self.decoder = nn.Linear(self.dict_dim,self.activation_dim,bias=False)
        self.decoder_bias = nn.Parameter(torch.empty(self.activation_dim))
        self._reset_parameters()

    def _reset_parameters(self):

        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        #random unit vectors

        dec_weight = torch.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0,keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self,x):
        
        x_enc = self.encoder(x-self.decoder_bias)

        pi_gate = x_enc + self.gate_bias                    #Gating Network
        f_gate = (pi_gate>0).to(self.encoder.weight.dtype)
  
        pi_mag = self.r_mag.exp()*x_enc + self.mag_bias     #Magnitude Network
        f_mag = nn.ReLU()(pi_mag) 

        f = f_gate*f_mag

        f = f * self.decoder.weight.norm(dim=0, keepdim=True) #Anthropic team was smoking

        return f

    def decode(self,f):

        f = f / self.decoder.weight.norm(dim=0, keepdim=True) #Anthropic team was smoking
        return self.decoder(f)+self.decoder_bias

    def forward(self,x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss
        return loss, x_reconstruct, acts, l2_loss,l1_loss

if __name__ == '__main__':
    sae = GatedAutoEncoder(cfg=config)
    d = sae(torch.ones([config['activation_dim']]))
    print()