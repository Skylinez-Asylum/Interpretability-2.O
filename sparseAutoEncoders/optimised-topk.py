import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Configuration
config = {
    'activation_dim': 768,
    'dict_dim': 16384,
    'l1_coeff': 3e-4,
    'batch_size': 512,
    'num_epochs': 20,
    'lr': 1e-4,
    'k': 5
}

# Dataset
class SAE_Dataset(Dataset):
    def __init__(self, num_samples=10000):
        self.data = torch.randn(num_samples, config['activation_dim'])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], idx

# Model
class TopKAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.k = cfg['k']
        
        # Initialize encoder and decoder
        self.encoder = nn.Linear(cfg['activation_dim'], cfg['dict_dim'], bias=True)
        self.decoder = nn.Linear(cfg['dict_dim'], cfg['activation_dim'], bias=False)
        
        # Initialize weights and biases
        nn.init.kaiming_normal_(self.encoder.weight)
        self.encoder.bias.data.zero_()
        self.decoder.weight.data = self.encoder.weight.data.clone().T # Weight Tieing Scheme
        
        self.b_dec = nn.Parameter(torch.zeros(cfg['activation_dim']))
        self.set_decoder_norm_to_unit_norm() # normalisation

    def encode(self, x, return_topk: bool = False):
        encoder_output = self.encoder(x - self.b_dec) # no relu used
        top_k_values, top_k_indices = encoder_output.topk(self.k, dim=-1)
        
        sparse_acts = torch.zeros_like(encoder_output)
        sparse_acts.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        
        if return_topk:
            return sparse_acts, top_k_values, top_k_indices
        return sparse_acts

    def decode(self, x):
        return self.decoder(x) + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        x_reconstruct = self.decode(acts)
        
        l2_loss = (x_reconstruct - x).pow(2).sum(-1).mean()
        l1_loss = self.cfg['l1_coeff'] * acts.abs().sum(-1).mean()
        
        total_loss = l2_loss + l1_loss
        return total_loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        # normalisation
        eps = torch.finfo(self.decoder.weight.dtype).eps
        norm = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= (norm + eps)

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        # common technique in optimization and training stability
        if self.decoder.weight.grad is None: return None
        W_dec = self.decoder.weight
        W_dec_normed = W_dec / (W_dec.norm(dim=0, keepdim=True) + 1e-8)
        W_dec_grad_proj = (W_dec.grad * W_dec_normed).sum(0, keepdim=True) * W_dec_normed
        self.decoder.weight.grad -= W_dec_grad_proj

def train_model(model, train_loader, config, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    for epoch in range(config['num_epochs']):
        model.train()
        total_loss = 0
        
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            
            loss, _, _, l2_loss, l1_loss = model(x)
            loss.backward()
            
            model.remove_parallel_component_of_grads()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping
            
            optimizer.step()
            model.set_decoder_norm_to_unit_norm() # normalise
            
            total_loss += loss.item()
            
            # To see batches
            # if batch_idx % 100 == 0:
            #     print(f'Batch {batch_idx}/{len(train_loader)}, '
            #           f'Loss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        print(f'Epoch: {epoch+1}, Avg Loss: {avg_loss:.6f}, '
              f'L2: {l2_loss:.6f}, L1: {l1_loss:.6f}, '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # if (epoch + 1) % 10 == 0:
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': avg_loss,
        #     }, f'checkpoint_epoch_{epoch+1}.pt')

def main():
    from torch.utils.data import Dataset, DataLoader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = SAE_Dataset()
    train_dataloader = DataLoader(dataset = dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)
    test_dataloader = DataLoader(dataset = dataset, batch_size = config['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize and train model
    model = TopKAutoEncoder(config).to(device)
    train_model(model, train_dataloader, config, device)

if __name__ == "__main__":
    main()