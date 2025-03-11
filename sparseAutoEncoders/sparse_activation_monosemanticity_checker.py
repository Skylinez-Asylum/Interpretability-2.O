import torch
import torch.nn as nn
import numpy as np
import os

from sae_jumprelu import JumpReluAutoEncoder  # Assuming this is your custom SAE class

# Optimized configuration
config = {
    'activation_dim': 768,
    'dict_dim': 16384*16,
    'l1_coeff': 3e-4,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create output directory for results
os.makedirs('neuron_activation_counts', exist_ok=True)

# Load the model
model = JumpReluAutoEncoder(cfg=config).to(device)
checkpoint_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/GPT2_jumprelu/checkpoint_epoch_200v4.pt'

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.6f}")

model.eval()

def process_activations(activations_path, batch_size=32, max_activations=None):
    """
    Process activations and count how many times each neuron is activated.
    
    Args:
        activations_path: Path to .npy file with activations
        batch_size: Number of activations to process at once
        max_activations: Maximum number of activations to process (None for all)
    
    Returns:
        neuron_activation_counts: Tensor with count of activations per neuron
    """
    # Get total number of activations
    total_shape = np.load(activations_path, mmap_mode='r').shape
    total_activations = total_shape[0] if max_activations is None else min(total_shape[0], max_activations)
    
    print(f"Processing {total_activations} activations with batch size {batch_size}")
    
    # Initialize counter for neuron activations
    neuron_activation_counts = torch.zeros(config['dict_dim'], dtype=torch.long, device=device)
    
    chunk_size = min(5000, total_activations)  # Process in chunks to manage memory
    
    for chunk_start in range(0, total_activations, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_activations)
        print(f"Processing chunk {chunk_start} to {chunk_end}")
        
        # Load chunk of activations
        activations_chunk = torch.from_numpy(
            np.load(activations_path, mmap_mode='r')[chunk_start:chunk_end]
        ).to(device)
        
        # Process in batches
        for batch_start in range(0, activations_chunk.shape[0], batch_size):
            batch_end = min(batch_start + batch_size, activations_chunk.shape[0])
            batch = activations_chunk[batch_start:batch_end]
            
            # Forward pass through the model
            with torch.no_grad():
                _, _, acts, _, _ = model(batch)
            
            # Count active neurons (acts > 0) across batch
            active_neurons = (acts > 0).long()  # Convert to 1/0 tensor
            batch_counts = active_neurons.sum(dim=0)  # Sum across batch dimension
            neuron_activation_counts += batch_counts
            
            # Clear memory
            del batch
            torch.cuda.empty_cache()
            
            if (batch_start + batch_end) % (batch_size * 10) == 0:
                print(f"Processed {chunk_start + batch_start + batch_end} activations")
        
        # Clear chunk memory
        del activations_chunk
        torch.cuda.empty_cache()
    
    return neuron_activation_counts

def save_results(neuron_counts, save_path='neuron_activation_counts/neuron_counts.txt'):
    """Save neuron activation counts to a text file."""
    counts_cpu = neuron_counts.cpu().numpy()
    with open(save_path, 'w') as f:
        f.write("Neuron Activation Counts:\n")
        for i, count in enumerate(counts_cpu):
            f.write(f"Neuron {i}: {count} times\n")
    print(f"Results saved to {save_path}")

# Main execution
if __name__ == "__main__":
    activations_path = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2/GPT2activations.npy'
    
    # Ask user for number of activations to process
    max_acts_input = input("Enter number of activations to process (or 'all' for all activations): ")
    max_activations = None if max_acts_input.lower() == 'all' else int(max_acts_input)
    
    # Process activations and get neuron counts
    neuron_counts = process_activations(
        activations_path,
        batch_size=32,
        max_activations=max_activations
    )
    
    # Print summary
    total_activations_processed = np.load(activations_path, mmap_mode='r').shape[0] if max_activations is None else min(np.load(activations_path, mmap_mode='r').shape[0], max_activations)
    print("\nSummary:")
    print(f"Total activations processed: {total_activations_processed}")
    print(f"Total neurons: {config['dict_dim']}")
    print(f"Most active neuron count: {neuron_counts.max().item()}")
    print(f"Least active neuron count: {neuron_counts.min().item()}")
    print(f"Average activations per neuron: {neuron_counts.float().mean().item():.2f}")
    
    # Save results
    print(neuron_counts)
    save_results(neuron_counts)
    
    # Ask for visualization
    visualize = input("Do you want to visualize the neuron activation counts? (y/n): ").lower() == 'y'
    if visualize:
        import matplotlib.pyplot as plt
        
        counts_np = neuron_counts.cpu().numpy()
        plt.figure(figsize=(12, 6))
        plt.hist(counts_np, bins=50, log=True)
        plt.title("Distribution of Neuron Activation Counts")
        plt.xlabel("Number of Activations")
        plt.ylabel("Number of Neurons (log scale)")
        plt.savefig('neuron_activation_counts/neuron_counts_histogram.png')
        plt.close()
        print("Histogram saved to neuron_activation_counts/neuron_counts_histogram.png")
    
    print("Processing complete!")