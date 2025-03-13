import numpy as np

# Define paths
input_file = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations.npy'
output_file = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations_scaled.npy'

# Load the original activations
activations = np.load(input_file)

# Scale the activations by dividing by 1000
scaled_activations = activations / 1000.0

# Save the scaled activations to a new file
np.save(output_file, scaled_activations)

# Optional: Verify the scaling
print(f"Original activations - Mean: {activations.mean():.6f}, Std: {activations.std():.6f}, Min: {activations.min():.6f}, Max: {activations.max():.6f}")
print(f"Scaled activations - Mean: {scaled_activations.mean():.6f}, Std: {scaled_activations.std():.6f}, Min: {scaled_activations.min():.6f}, Max: {scaled_activations.max():.6f}")
print(f"Scaled activations saved to: {output_file}")