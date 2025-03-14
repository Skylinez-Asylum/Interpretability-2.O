'''
You can give a token and this code will run its activation through the SAE of your choice and will give the neurons that activated for each of those tokens.
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sae_jumprelu import JumpReluAutoEncoder
from activationManager import ActivationManager

# Assume JumpReluAutoEncoder and ActivationManager classes are defined as in the query

# Load the model
model_path = Path("/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/CustomFT_jumprelu/model_v12_500-best.pt")
# state_dict = torch.load(model_path, map_location='cpu')
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
activation_dim = state_dict['W_enc'].shape[0]
dict_dim = state_dict['W_enc'].shape[1]
cfg = {'activation_dim': activation_dim, 'dict_dim': dict_dim, 'l1_coeff': 0.01, 'dropout_rate': 0.1}
model = JumpReluAutoEncoder(cfg)
model.load_state_dict(state_dict)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Initialize ActivationManager
manager = ActivationManager("/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2FT/activations.pkl")

# Define the function
def get_activated_neurons_for_token(token, model, manager, device):
    act_list = manager.get_activations(token)
    if not act_list:
        print(f"No activations found for token '{token}'.")
        return {}
    act_array = np.stack(act_list)
    act_tensor = torch.from_numpy(act_array).float().to(device)
    with torch.no_grad():
        acts_tensor = model.encode(act_tensor)
    activation_frequency = (acts_tensor > 0).float().mean(dim=0).cpu().numpy()
    activated_neurons = np.where(activation_frequency > 0)[0]
    return {int(neuron): float(activation_frequency[neuron]) for neuron in activated_neurons}

# Analyze a token
token = ' is'
neuron_info = get_activated_neurons_for_token(token, model, manager, device)
print(f"Neurons activated by token '{token}':")
for neuron_idx, freq in sorted(neuron_info.items()):
    print(f"Neuron {neuron_idx}: Activated in {freq:.2%} of instances")