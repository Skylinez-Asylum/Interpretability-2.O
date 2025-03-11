'''
This code will load SAE save and will tell you how many neurons are being active for each activation
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_jumprelu import JumpReluAutoEncoder
import matplotlib.pyplot as plt
import numpy as np
import os

# Optimized configuration
config = {
    'activation_dim': 768,
    'dict_dim': 16384*16,
    'l1_coeff': 3e-4,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory for visualizations
os.makedirs('activation_visualizations', exist_ok=True)

# Load the model
model = JumpReluAutoEncoder(cfg=config).to(device)
checkpoint_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/GPT2_jumprelu/checkpoint_epoch_200v4.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

model.eval()

def visualize_activation(acts, idx, num_active, save_dir='activation_visualizations'):
    # Move to CPU and convert to numpy to free GPU memory
    acts_np = acts.squeeze(0).cpu().numpy()
    
    # Calculate reshaping dimensions
    total_neurons = acts_np.shape[0]
    grid_size = int(np.sqrt(total_neurons))
    if grid_size * grid_size != total_neurons:
        # If not a perfect square, find closest factors
        factors = []
        for i in range(1, int(np.sqrt(total_neurons)) + 1):
            if total_neurons % i == 0:
                factors.append((i, total_neurons // i))
        factors.sort(key=lambda x: abs(x[0] - x[1]))
        grid_h, grid_w = factors[0]
    else:
        grid_h = grid_w = grid_size
    
    acts_2d = acts_np.reshape(grid_h, grid_w)
    
    # Use a simpler visualization approach
    plt.figure(figsize=(8, 8))
    plt.imshow(acts_2d > 0, cmap='RdYlGn', interpolation='nearest')
    plt.title(f"Activation {idx} - {num_active} Active Neurons")
    plt.axis('off')
    plt.savefig(f"{save_dir}/activation_{idx}_{num_active}_neurons.png")
    plt.close()  # Close the figure to free memory

def process_activations(activations_path, batch_size=32, top_n=20, max_activations=None):
    # Get total number of activations without loading everything
    total_shape = np.load(activations_path, mmap_mode='r').shape
    total_activations = total_shape[0] if max_activations is None else min(total_shape[0], max_activations)
    
    print(f"Processing {total_activations} activations with batch size {batch_size}")
    
    results = []
    chunk_size = min(10000, total_activations)  # Load at most 10k at a time to save memory
    
    for chunk_start in range(0, total_activations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_activations)
        print(f"Loading activations chunk {chunk_start} to {chunk_end}")
        
        # Load a chunk of activations
        all_activations_chunk = torch.from_numpy(
            np.load(activations_path, mmap_mode='r')[chunk_start:chunk_end]
        ).to(device)
        
        # Process this chunk in batches
        for batch_start in range(0, all_activations_chunk.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, all_activations_chunk.shape[0])
            batch = all_activations_chunk[batch_start:batch_end]
            
            # Forward pass through the model
            with torch.no_grad():
                _, _, acts, _, _ = model(batch)
            
            # Process each activation in the batch
            for i in range(acts.shape[0]):
                idx = chunk_start + batch_start + i
                activation_acts = acts[i:i+1]
                active_neurons = (activation_acts > 0).squeeze(0)
                active_count = active_neurons.sum().item()
                
                # Only store if potentially in top_n
                if len(results) < top_n or active_count < results[-1][1]:
                    results.append((idx, active_count, activation_acts.detach()))
                    # Keep only top_n results sorted by active neuron count
                    results.sort(key=lambda x: x[1])
                    if len(results) > top_n:
                        # Remove the item with most active neurons, also free memory
                        _, _, removed_acts = results.pop()
                        del removed_acts
            
            # Clear batch to free memory
            del batch
            torch.cuda.empty_cache()
            
            if (batch_start + batch_end) % (batch_size * 10) == 0:
                print(f"Processed {chunk_start + batch_start + batch_end} activations")
        
        # Clear chunk to free memory
        del all_activations_chunk
        torch.cuda.empty_cache()
    
    return results

# Main execution
if __name__ == "__main__":
    activations_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2/GPT2activations.npy'
    
    results_sorted = process_activations(
        activations_path, 
        batch_size=32,
        top_n=20,
        max_activations=None # give the number of activations to take(None takes all)
    )
    
    # Display results
    print("\nTop Activations with Minimum Active Neurons:")
    for rank, (idx, num_active, _) in enumerate(results_sorted, 10):
        print(f"Rank {rank}: Index {idx}, Active Neurons: {num_active}")
    
    # Ask user if they want to visualize
    visualize = input("Do you want to visualize these activations? (y/n): ").lower() == 'y'
    if visualize:
        for rank, (idx, num_active, acts) in enumerate(results_sorted, 1):
            print(f"Visualizing {rank}/{len(results_sorted)}: Index {idx}")
            visualize_activation(acts, idx, num_active)
    
    print(f"Done! Visualizations saved to ./activation_visualizations/")
