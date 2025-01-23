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
        x_enc = x @ self.W_enc + self.b_enc
        
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(x_enc.dtype)
        
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = F.relu(pi_mag)
        
        acts = f_gate * f_mag
        acts = acts * self.W_dec.norm(dim=1)
        
        return acts

    def decode(self, f):
        
        norm = self.W_dec.norm(dim=1, keepdim=True).t() 
        f = f / norm 
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

    def resample_dead_neurons(self, optimizer, dataset):
        if self.step_counter < 10000 or self.step_counter % 5000 != 0:
            return

        dead_threshold = 0.01  # Consider neurons dead if activity is very low
        normalized_activity = self.neuron_activity / self.step_counter
        dead_neurons = (normalized_activity < dead_threshold).nonzero(as_tuple=True)[0]

        if len(dead_neurons) == 0:
            return

        subset_size = min(len(dataset), 1000000)
        subset = torch.utils.data.Subset(dataset, torch.randperm(len(dataset))[:subset_size])
        
        errors = []
        with torch.no_grad():
            for input_data in subset:
                _, x_reconstruct, _, _, _ = self(input_data.unsqueeze(0))
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