import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_relu import ReluAutoEncoder
import matplotlib.pyplot as plt
import numpy as np

# Configuration (same as your training setup)
config = {
    'activation_dim': 768,
    'dict_dim': 16384,
    'l1_coeff': 3e-4,
    'batch_size': 1280,
    'num_epochs': 400,
    'lr': 5e-4,
    'gradient_clip_val': 1.0,
    'checkpoint_frequency': 100, 
    'dropout_rate': 0.1,
    'weight_decay': 1e-5,
    'gradient_clip_val': 0.5
}

device = torch.device('cuda')

# Load the model
model = ReluAutoEncoder(cfg=config).to(device)
checkpoint_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/checkpoint_epoch_400.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

model.eval()

def process_random_activation(activation, model, device='cuda'):
    # Forward pass
    with torch.no_grad():
        loss, x_reconstruct, acts, l2_loss, l1_loss = model(activation)
    
    # Identify active neurons (where acts > 0)
    active_neurons = (acts > 0).squeeze(0)  # Shape: [16384]
    active_indices = active_neurons.nonzero(as_tuple=True)[0]  # Indices of active neurons
    active_values = acts.squeeze(0)[active_indices]  # Activation values for active neurons

    print(f"\nNumber of Active Neurons: {len(active_indices)} out of {config['dict_dim']}")

    # Prepare activation visualization
    acts_np = acts.squeeze(0).cpu().numpy()  # Shape: [16384]
    grid_size = int(np.sqrt(config['dict_dim']))  # 128 for 16384 (128x128)
    acts_2d = acts_np.reshape(grid_size, grid_size)  # Reshape to 128x128

    # Create a color map: green for positive, red for zero/negative
    color_map = np.zeros((grid_size, grid_size, 3))  # RGB
    color_map[acts_2d > 0] = [0, 1, 0]  # Green for positive
    color_map[acts_2d == 0] = [1, 0, 0]  # Red for zero/negative
    color_map[acts_2d < 0] = [0, 0, 1] # Blue never going to happen

    # Plot the image
    plt.figure(figsize=(8, 8))
    plt.imshow(color_map, interpolation='nearest')
    plt.title('Neuron Activations (Green: >0, Red: ≤0)')
    plt.axis('off')  # Hide axes
    plt.show()

    return x_reconstruct, acts

# Run the function
activation = torch.from_numpy(np.load('activations/GPT2/GPT2activations.npy')[0]).to(device)
reconstruction, activations = process_random_activation(activation, model)