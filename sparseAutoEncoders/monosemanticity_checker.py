import torch
import torch.nn as nn
import numpy as np
import os
from sae_jumprelu import JumpReluAutoEncoder

# --- Adjust these
checkpoint_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/CustomFT_jumprelu/model_v10_5.pt'
config = {
    'activation_dim': 768,
    'dict_dim': 16384,
    'l1_coeff': 3e-4,
}
activations_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations_scaled.npy'
image_shape = (128, 128)
overwrite = True # switch to adjust activation count
# --- Adjust these

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs('neuron_activation_counts', exist_ok=True)

# Load the model
model = JumpReluAutoEncoder(cfg=config).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")
model.eval()

def process_activations(activations_path, batch_size=32, max_activations=None):
    """
    Process activations, count neuron activations, and track number of active neurons per activation.
    """
    total_shape = np.load(activations_path, mmap_mode='r').shape
    total_activations = total_shape[0] if max_activations is None else min(total_shape[0], max_activations)
    print(f"Processing {total_activations} activations with batch size {batch_size}")
    
    neuron_activation_counts = torch.zeros(config['dict_dim'], dtype=torch.long, device=device)
    active_neurons_per_activation = []  # List to store number of active neurons per activation
    
    chunk_size = min(100000, total_activations)
    
    for chunk_start in range(0, total_activations, chunk_size): 
        chunk_end = min(chunk_start + chunk_size, total_activations)
        print(f"Processing chunk {chunk_start} to {chunk_end}")
        
        activations_chunk = torch.tensor(np.load(activations_path, mmap_mode='r')[chunk_start:chunk_end]).to(device)
        
        for batch_start in range(0, activations_chunk.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, activations_chunk.shape[0])
            batch = activations_chunk[batch_start:batch_end]
            
            with torch.no_grad():
                _, _, acts, _, _ = model(batch)
            
            # Count active neurons (acts > 0)
            active_neurons = (acts > 0).long()
            batch_counts = active_neurons.sum(dim=0)  # Total activations per neuron in batch
            neuron_activation_counts += batch_counts
            
            # Count number of active neurons per activation in this batch
            num_active_per_activation = active_neurons.sum(dim=1)  # Sum across neurons for each activation
            active_neurons_per_activation.extend(num_active_per_activation.cpu().tolist())
            
            del batch
            torch.cuda.empty_cache()
            
            if (batch_start + batch_end) % (batch_size * 10) == 0:
                print(f"Processed {chunk_start + batch_start + batch_end} activations")
        
        del activations_chunk
        torch.cuda.empty_cache()
    
    return neuron_activation_counts, active_neurons_per_activation

def save_results(neuron_counts, active_neurons_per_activation, save_path='neuron_activation_counts/neuron_counts.txt'):
    """Save neuron activation counts and summary of active neurons per activation."""
    counts_cpu = neuron_counts.cpu().numpy()
    with open(save_path, 'w') as f:
        f.write("Neuron Activation Counts:\n")
        for i, count in enumerate(counts_cpu):
            f.write(f"Neuron {i}: {count} times\n")
        
        # Calculate frequency of number of active neurons per activation
        unique_active_counts, frequencies = np.unique(active_neurons_per_activation, return_counts=True)
        f.write("\nNumber of Active Neurons per Activation Summary:\n")
        f.write("Number of Active Neurons -> Number of Activations\n")
        for count, freq in zip(unique_active_counts, frequencies):
            f.write(f"{count} -> {freq}\n")
    
    print(f"Results saved to {save_path}")

# Main execution
if __name__ == "__main__":
    if overwrite:
        max_activations = None
    else:
        max_acts_input = input("Enter number of activations to process (Press enter to choose all): ")
        max_activations = None if max_acts_input.lower() == '' else int(max_acts_input)
    
    neuron_counts, active_neurons_per_activation = process_activations(
        activations_path,
        batch_size=32,
        max_activations=max_activations
    )
    
    total_activations_processed = np.load(activations_path, mmap_mode='r').shape[0] if max_activations is None else min(np.load(activations_path, mmap_mode='r').shape[0], max_activations)
    print("\nSummary:")
    print(f"Total activations processed: {total_activations_processed}")
    print(f"Total neurons: {config['dict_dim']}")
    print(f"Most active neuron count: {neuron_counts.max().item()}")
    print(f"Least active neuron count: {neuron_counts.min().item()}")
    print(f"Average activations per neuron: {neuron_counts.float().mean().item():.2f}")
    
    save_results(neuron_counts, active_neurons_per_activation)
    if not overwrite: 
        visualize = input("Do you want to visualize the neuron activation counts? (y/n): ").lower() == 'y'
    else: visualize = True
    if visualize:
        import matplotlib.pyplot as plt
        
        counts_np = neuron_counts.cpu().numpy()
        plt.figure(figsize=(24, 6))
        plt.hist(counts_np, bins=1000)
        plt.title("Distribution of Neuron Activation Counts")
        plt.xlabel("Number of Activations")
        plt.ylabel("Number of Neurons (log scale)")
        plt.savefig('neuron_activation_counts/neuron_counts_histogram.png')
        plt.close()
        print("Histogram saved to neuron_activation_counts/neuron_counts_histogram.png")
    
    print("Processing complete!")