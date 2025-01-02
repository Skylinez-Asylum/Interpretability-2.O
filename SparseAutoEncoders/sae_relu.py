import torch
import torch.nn as nn
import torch.nn.functional as F
import random



class ReluAutoEncoder(nn.Module):

    def __init__(self,cfg):
        super().__init__()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.cfg = cfg
        self.l1_coeff = cfg['l1_coeff']
        #weights
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg['activation_dim'],cfg['dict_dim'])))
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(cfg['dict_dim'],cfg['activation_dim'])))
        #bias
        self.b_enc = nn.Parameter(torch.zeros(cfg['dict_dim']))
        self.b_dec = nn.Parameter(torch.zeros(cfg['activation_dim']))
        #idk y i should normalise tell me if you know 
        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.neuron_activity = torch.zeros(cfg['dict_dim']).to(device)
        self.step_counter = 0

    def forward(self,x):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent@self.W_enc + self.b_enc)
        x_reconstruct = acts@self.W_dec + self.b_dec
        l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)
        l1_loss = self.l1_coeff * (acts.float().abs().sum())
        loss = l2_loss + l1_loss

        # Update neuron activity
        self.neuron_activity += (acts > 0).float().sum(0)
        self.step_counter += 1

        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @torch.no_grad()
    def remove_parallel_component_of_grads(self):
        #IDK WHY 
        W_dec_normed = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)
        W_dec_grad_proj = (self.W_dec.grad * W_dec_normed).sum(-1, keepdim=True) * W_dec_normed
        self.W_dec.grad -= W_dec_grad_proj  
    

    def resample_dead_neurons(self, optimizer, dataset):
        #Generate Code Need To Test My code is brrr!!!!

        if self.step_counter not in [25000, 50000, 75000, 100000]:
            return

        # 1. Identify dead neurons
        dead_neurons = (self.neuron_activity == 0).nonzero(as_tuple=True)[0]

        if len(dead_neurons) == 0:
            return

        # 2. Compute loss on a subset of inputs
        subset_size = min(819200, len(dataset))
        subset = random.sample(dataset, subset_size)
        losses = []
        for input_data in subset:
            loss, _, _, _, _ = self.forward(input_data.unsqueeze(0))
            losses.append(loss.item())

        # 3. Assign probabilities proportional to squared loss
        probs = torch.tensor(losses) ** 2
        probs /= probs.sum()

        # 4-6. Resample dead neurons
        for neuron_idx in dead_neurons:
            # Sample an input
            input_idx = torch.multinomial(probs, 1).item()
            input_vector = subset[input_idx]

            # Set dictionary vector
            self.W_dec.data[neuron_idx] = F.normalize(input_vector, dim=0)

            # Set encoder vector
            avg_norm = self.W_enc.data.norm(dim=0).mean().item()
            self.W_enc.data[:, neuron_idx] = F.normalize(input_vector, dim=0) * (avg_norm * 0.2)
            self.b_enc.data[neuron_idx] = 0

            # Reset optimizer state for modified weights and biases
            if optimizer.state.get(self.W_enc) is not None:
                optimizer.state[self.W_enc]['exp_avg'][:, neuron_idx] = 0
                optimizer.state[self.W_enc]['exp_avg_sq'][:, neuron_idx] = 0
            if optimizer.state.get(self.W_dec) is not None:
                optimizer.state[self.W_dec]['exp_avg'][neuron_idx] = 0
                optimizer.state[self.W_dec]['exp_avg_sq'][neuron_idx] = 0
            if optimizer.state.get(self.b_enc) is not None:
                optimizer.state[self.b_enc]['exp_avg'][neuron_idx] = 0
                optimizer.state[self.b_enc]['exp_avg_sq'][neuron_idx] = 0

        # Reset neuron activity counter
        self.neuron_activity.zero_()
        self.step_counter = 0

if __name__ == '__main__':
    config = {
    'activation_dim':768,
    'dict_dim':16384,
    'l1_coeff':3e-4,
    'batch_size': 128,
    'num_epochs': 1000,
    'lr':1e-4
    }

    sae = ReluAutoEncoder(cfg=config)
    d = sae(torch.ones([config['activation_dim']]))



