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
            x_cent = x_cent + torch.randn_like(x_cent) * 0.01
            
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
        # print(self.neuron_activity)
        self.step_counter += 1

        return loss, x_reconstruct, acts, l2_loss, l1_loss


    def resample_dead_neurons(self, optimizer, dataset):
        print('resampling')
        self.neuron_activity = torch.zeros_like(self.neuron_activity)
    
        # Identify dead neurons
        dead_neurons = (self.neuron_activity == 0).nonzero(as_tuple=True)[0]
        if len(dead_neurons) == 0:
            return

        # Convert dataset to a list if it's not already a sequence
        if hasattr(dataset, 'tensors'):  # TensorDataset
            data_list = [x for x in dataset.tensors[0]]
        elif isinstance(dataset, torch.utils.data.Dataset):
            data_list = [dataset[i][0] for i in range(len(dataset))]
        else:
            data_list = list(dataset)
        
        # Compute loss on subset of inputs
        subset_size = min(100, len(data_list))
        subset = random.sample(data_list, subset_size)
        losses = []
        
        for input_data in subset:
            if not isinstance(input_data, torch.Tensor):
                continue
            input_tensor = input_data.unsqueeze(0) if input_data.dim() == 1 else input_data
            loss = self.forward(input_tensor.to('cuda'))[0]
            losses.append(loss.item())
        
        if not losses:
            return
            
        # Assign probabilities proportional to squared loss
        probs = torch.tensor(losses) ** 2
        probs /= probs.sum()
        
        # Resample dead neurons
        for neuron_idx in dead_neurons:
            # Sample an input
            input_idx = torch.multinomial(probs, 1).item()
            input_vector = subset[input_idx]
            
            # Set dictionary vector
            self.W_dec.data[neuron_idx] = F.normalize(input_vector, dim=0)
            
            # Set encoder vector with scaled normalization
            avg_norm = self.W_enc.data.norm(dim=0).mean().item()
            self.W_enc.data[:, neuron_idx] = F.normalize(input_vector, dim=0) * (avg_norm * 0.2)
            self.b_enc.data[neuron_idx] = 0
            
            # Reset optimizer state
            for param_name in ['W_enc', 'W_dec', 'b_enc']:
                param = getattr(self, param_name)
                if optimizer.state.get(param) is not None:
                    if param_name == 'W_enc':
                        optimizer.state[param]['exp_avg'][:, neuron_idx] = 0
                        optimizer.state[param]['exp_avg_sq'][:, neuron_idx] = 0
                    elif param_name == 'W_dec':
                        optimizer.state[param]['exp_avg'][neuron_idx] = 0
                        optimizer.state[param]['exp_avg_sq'][neuron_idx] = 0
                    else:  # b_enc
                        optimizer.state[param]['exp_avg'][neuron_idx] = 0
                        optimizer.state[param]['exp_avg_sq'][neuron_idx] = 0
        
        # Reset counters
        self.neuron_activity.zero_()
        self.step_counter = 0


    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        #IDK WHY 
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj  


    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


