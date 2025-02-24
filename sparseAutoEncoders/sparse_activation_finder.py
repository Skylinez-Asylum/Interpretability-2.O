import torch
import torch.nn as nn
import torch.nn.functional as F
# from sae_relu import ReluAutoEncoder
from sae_jumprelu import JumpReluAutoEncoder
import matplotlib.pyplot as plt
import numpy as np

# Configuration (same as your training setup)
config = {
    'activation_dim': 768,
    'dict_dim': 16384*2,
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
model = JumpReluAutoEncoder(cfg=config).to(device)
checkpoint_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/GPT2_jumprelu/checkpoint_epoch_500v3.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

model.eval()

# Function to process an activation and return number of active neurons and activations
def get_active_neurons(activation, model, device='cuda'):
    with torch.no_grad():
        _, _, acts, _, _ = model(activation)
    active_neurons = (acts > 0).squeeze(0)  # Shape: [16384]
    active_indices = active_neurons.nonzero(as_tuple=True)[0]
    return len(active_indices), acts

# Function to visualize activations
def visualize_activation(acts, title):
    acts_np = acts.squeeze(0).cpu().numpy()  # Shape: [16384]
    grid_size = int(np.sqrt(config['dict_dim']))  # 128 for 16384 (128x128)
    acts_2d = acts_np.reshape(128, 128*2)  # Reshape to 128x128

    # Create a color map: green for positive, red for zero/negative
    color_map = np.zeros((128, 128*2, 3))  # RGB
    color_map[acts_2d > 0] = [0, 1, 0]  # Green for positive
    color_map[acts_2d == 0] = [1, 0, 0]  # Red for zero/negative

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(color_map, interpolation='nearest')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Load activations
activations_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2/GPT2activations.npy'
all_activations = torch.from_numpy(np.load(activations_path)).to(device)  # Shape: [148640, 768]
print(f"Loaded {all_activations.shape[0]} activations")

# Process all activations and count active neurons
results = []
for idx in range(all_activations.shape[0]):
    activation = all_activations[idx:idx+1]  # Shape: [1, 768]
    num_active, acts = get_active_neurons(activation, model, device)
    results.append((idx, num_active, acts))
    if idx % 1000 == 0:
        print(f"Processed {idx} activations")

# Sort by number of active neurons (ascending) and get top 20
results_sorted = sorted(results, key=lambda x: x[1])[:20]

# Display top 20 results
print("\nTop 20 Activations with Minimum Active Neurons:")
for rank, (idx, num_active, acts) in enumerate(results_sorted, 1):
    print(f"Rank {rank}: Index {idx}, Active Neurons: {num_active}")
    visualize_activation(acts, f"Activation {idx} - {num_active} Active Neurons")

# To future me: add the code to find the token whose activation caused this. 