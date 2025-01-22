import torch
import torch.nn as nn
import torch.nn.functional as F
import random



# class ReluAutoEncoder(nn.Module):

#     def __init__(self,cfg):
#         super().__init__()
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.cfg = cfg
#         self.l1_coeff = cfg['l1_coeff']
#         #weights
#         self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg['activation_dim'],cfg['dict_dim'])))
#         self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg['dict_dim'],cfg['activation_dim'])))
#         #bias
#         self.b_enc = nn.Parameter(torch.zeros(cfg['dict_dim']))
#         self.b_dec = nn.Parameter(torch.zeros(cfg['activation_dim']))
#         #idk y i should normalise tell me if you know 
#         self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

#         self.neuron_activity = torch.zeros(cfg['dict_dim']).to(device)
#         self.step_counter = 0

#     def forward(self,x):
#         x_cent = x - self.b_dec
#         acts = F.relu(x_cent@self.W_enc + self.b_enc)
#         x_reconstruct = acts@self.W_dec + self.b_dec
#         l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
#         l1_loss = self.l1_coeff * (acts.float().abs().sum())
#         loss = l2_loss + l1_loss

#         # Update neuron activity
#         self.neuron_activity += (acts > 0).float().sum(0)
#         self.step_counter += 1

#         return loss, x_reconstruct, acts, l2_loss, l1_loss

#     @torch.no_grad()
#     def remove_parallel_component_of_grads(self):
#         #IDK WHY 
#         W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
#         W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
#         self.W_dec.grad -= W_dec_grad_proj  
    

#     def resample_dead_neurons(self, optimizer, dataset):
#         #Generate Code Need To Test My code is brrr!!!!

#         if self.step_counter not in [25000, 50000, 75000, 100000]:
#             return

#         # 1. Identify dead neurons
#         dead_neurons = (self.neuron_activity == 0).nonzero(as_tuple=True)[0]

#         if len(dead_neurons) == 0:
#             return

#         # 2. Compute loss on a subset of inputs
#         subset_size = min(819200, len(dataset))
#         subset = random.sample(dataset, subset_size)
#         losses = []
#         for input_data in subset:
#             loss, _, _, _, _ = self.forward(input_data.unsqueeze(0))
#             losses.append(loss.item())

#         # 3. Assign probabilities proportional to squared loss
#         probs = torch.tensor(losses) ** 2
#         probs /= probs.sum()

#         # 4-6. Resample dead neurons
#         for neuron_idx in dead_neurons:
#             # Sample an input
#             input_idx = torch.multinomial(probs, 1).item()
#             input_vector = subset[input_idx]

#             # Set dictionary vector
#             self.W_dec.data[neuron_idx] = F.normalize(input_vector, dim=0)

#             # Set encoder vector
#             avg_norm = self.W_enc.data.norm(dim=0).mean().item()
#             self.W_enc.data[:, neuron_idx] = F.normalize(input_vector, dim=0) * (avg_norm * 0.2)
#             self.b_enc.data[neuron_idx] = 0

#             # Reset optimizer state for modified weights and biases
#             if optimizer.state.get(self.W_enc) is not None:
#                 optimizer.state[self.W_enc]['exp_avg'][:, neuron_idx] = 0
#                 optimizer.state[self.W_enc]['exp_avg_sq'][:, neuron_idx] = 0
#             if optimizer.state.get(self.W_dec) is not None:
#                 optimizer.state[self.W_dec]['exp_avg'][neuron_idx] = 0
#                 optimizer.state[self.W_dec]['exp_avg_sq'][neuron_idx] = 0
#             if optimizer.state.get(self.b_enc) is not None:
#                 optimizer.state[self.b_enc]['exp_avg'][neuron_idx] = 0
#                 optimizer.state[self.b_enc]['exp_avg_sq'][neuron_idx] = 0

#         # Reset neuron activity counter
#         self.neuron_activity.zero_()
#         self.step_counter = 0

# if __name__ == '__main__':
#     config = {
#     'activation_dim':768,
#     'dict_dim':16384,
#     'l1_coeff':3e-4,
#     'batch_size': 128,
#     'num_epochs': 1000,
#     'lr':1e-4
#     }

#     sae = ReluAutoEncoder(cfg=config)
#     d = sae(torch.ones([config['activation_dim']]))
#     print(d)



class ReluAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.l1_coeff = cfg['l1_coeff']
        
        # Add dropout for regularization
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
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * acts.float().abs().sum()
        
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

    def resample_dead_neurons(self, optimizer, dataset):
        # Check more frequently for dead neurons
        if self.step_counter < 10000 or self.step_counter % 5000 != 0:
            return

        dead_threshold = 0.01  # Consider neurons dead if activity is very low
        normalized_activity = self.neuron_activity / self.step_counter
        dead_neurons = (normalized_activity < dead_threshold).nonzero(as_tuple=True)[0]

        if len(dead_neurons) == 0:
            return

        # Use more data points for resampling
        subset_size = min(len(dataset), 1000000)
        subset = random.sample(dataset, subset_size)
        
        # Compute reconstruction error for each sample
        errors = []
        with torch.no_grad():
            for input_data in subset:
                _, x_reconstruct, _, _, _ = self.forward(input_data.unsqueeze(0))
                error = (x_reconstruct - input_data).pow(2).sum()
                errors.append(error.item())

        # Sample from points with high reconstruction error
        probs = torch.tensor(errors)
        probs = probs / probs.sum()

        for neuron_idx in dead_neurons:
            # Sample multiple inputs and use their average
            num_samples = 5
            selected_indices = torch.multinomial(probs, num_samples)
            selected_inputs = torch.stack([subset[idx] for idx in selected_indices])
            input_vector = selected_inputs.mean(0)

            # Update dictionary element
            self.W_dec.data[neuron_idx] = F.normalize(input_vector, dim=0)

            # Update encoder weights with noise for exploration
            avg_norm = self.W_enc.data.norm(dim=0).mean().item()
            noise = torch.randn_like(input_vector) * 0.1
            self.W_enc.data[:, neuron_idx] = F.normalize(input_vector + noise, dim=0) * avg_norm

            # Reset optimizer state
            for param in [self.W_enc, self.W_dec, self.b_enc]:
                if param in optimizer.state:
                    for key in optimizer.state[param]:
                        if torch.is_tensor(optimizer.state[param][key]):
                            if key == 'exp_avg':
                                optimizer.state[param][key].zero_()
                            elif key == 'exp_avg_sq':
                                optimizer.state[param][key].fill_(optimizer.defaults['eps'])

        self.neuron_activity.zero_()
        self.step_counter = 0

    @torch.no_grad()
    def normalize_decoder_weights(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)