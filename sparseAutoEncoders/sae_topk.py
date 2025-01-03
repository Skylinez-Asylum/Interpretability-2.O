import torch
import torch.nn.functional as F
import torch.nn as nn


config = {
    'activation_dim':768,
    'dict_dim':16384,
    'l1_coeff':3e-4,
    'batch_size': 32,
    'num_epochs': 200,
    'lr':1e-4,
    'k': 5
}

class TopKAutoEncoder(nn.Module):
    
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.k = cfg['k']

        self.encoder = nn.Linear(self.cfg['activation_dim'],self.cfg['dict_dim'])
        self.encoder.bias.data.zero_() #Confused to use bias in encoder or not

        self.decoder = nn.Linear(self.cfg['dict_dim'],self.cfg['activation_dim'],bias=False)
        self.decoder.weight.data = self.encoder.weight.data.clone().T # Weight Tieing Scheme
        #self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(self.cfg['activation_dim']))

    def encode(self,x,return_topk: bool = False):
        post_relu = F.relu(self.encoder(x-self.b_dec)) # i think relu shouln't be used what if all ouputs are negative?
        post_topk = post_relu.topk(self.k,sorted=False,dim=-1)

        tops_acts,top_indices = post_topk.values,post_topk.indices
        # print(top_indices)

        buffer = torch.zeros_like(post_relu)
        encoder_acts = buffer.scatter_(dim=-1,index=top_indices, src=tops_acts)
        
        if return_topk:
            return encoder_acts, tops_acts, top_indices
        else:
            return encoder_acts
    
    def decode(self,x):
        return self.decoder(x)+self.b_dec

    def forward(self,x):
        acts = self.encode(x)
        # t(acts)
        x_reconstruct = self.decode(acts)
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        print(l2_loss)
        l1_loss = torch.tensor(0) # sample to fill the spot
        loss = l2_loss+l1_loss
        return loss, x_reconstruct, acts, l2_loss,l1_loss

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

        
    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        #IDK WHY 
        W_dec_normed = self.decoder / self.decoder.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.decoder.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.decoder.grad -= W_dec_grad_proj 

if __name__ == '__main__':
    sae = TopKAutoEncoder(cfg=config)
    d = sae(torch.ones([config['activation_dim']]))