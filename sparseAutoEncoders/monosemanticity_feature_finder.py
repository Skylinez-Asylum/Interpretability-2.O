'''
This code finds the tokens for which a given SAE activates one neuron, not sure if this guy is properly utilising cuda, but this taks 2mins, so will optimise later
'''

import torch
import numpy as np
from sae_jumprelu import JumpReluAutoEncoder
from activationManager import ActivationManager
from tqdm import tqdm
filename = 'model_v5_10.pt'
def analyze_neuron_activations(
    checkpoint_path: str = f'/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/CustomFT_jumprelu/{filename}',
   
    activations_storage: str = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations.pkl',
    output_path: str = f'neuron_activations_without_zero_neuron_{filename}.txt',
    batch_size: int = 32,
    normalization_factor: float = 1000.0
):
    """
    Analyze all activations through the SAE and record how many activations activated each neuron,
    along with the unique tokens that caused those activations.

    Args:
        checkpoint_path (str): Path to the trained SAE model checkpoint.
        activations_storage (str): Path to the stored activations.
        output_path (str): Path to save the results.
        batch_size (int): Batch size for processing activations.
        normalization_factor (float): Factor to divide activations by for normalization (default: 1.0, no normalization).
    """
    # **Device Setup**
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # **Load the SAE Model**
    config = {
        'activation_dim': 768,  # Input dimension of activations
        'dict_dim': 16384,     # Number of neurons in the SAE dictionary
        'l1_coeff': 3e-4,      # L1 regularization coefficient (not used here but part of config)
    }
    model = JumpReluAutoEncoder(cfg=config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}, epoch {checkpoint['epoch']}")

    # **Load Activations**
    act_manager = ActivationManager(storage_path=activations_storage)
    all_tokens = list(act_manager.activations.keys())
    print(all_tokens)
    print(len(all_tokens))
    print(f"Total unique tokens: {len(all_tokens)}")

    # **Initialize Tracking Structures**
    neuron_counts = [0] * config['dict_dim']  # Count of activations per neuron
    neuron_tokens = [set() for _ in range(config['dict_dim'])]  # Unique tokens per neuron

    # **Process Each Token's Activations**
    for token in tqdm(all_tokens, desc="Processing tokens"):
        activations = act_manager.get_activations(token)
        if not activations:
            continue

        # Convert activations to tensor and apply normalization
        activations_array = np.array(activations)
        normalized_activations = activations_array / normalization_factor
        activations_tensor = torch.tensor(normalized_activations).to(device)

        # Process in batches to manage memory
        for batch_start in range(0, len(activations), batch_size):
            batch_end = min(batch_start + batch_size, len(activations))
            batch = activations_tensor[batch_start:batch_end]

            # Pass through the SAE
            with torch.no_grad():
                _, _, acts, _, _ = model(batch)

            # Update counts and tokens for active neurons
            for i in range(acts.shape[0]):
                active_neurons = torch.where(acts[i] > 0)[0].cpu().numpy()
                for j in active_neurons:
                    neuron_counts[j] += 1
                    neuron_tokens[j].add(token)

            # Clear memory
            del batch
            torch.cuda.empty_cache()

    # **Write Results to File**
    with open(output_path, 'w') as f:
        for j in range(config['dict_dim']):
            count = neuron_counts[j]
            if count == 0:
                continue
            tokens = sorted(list(neuron_tokens[j]))  # Sort tokens for consistency
            tokens_str = ', '.join(tokens)
            line = f"neuron {j}: {count} activations [{tokens_str}]\n"
            f.write(line)

    print(f"Results saved to {output_path}")

    # **Summary Statistics**
    total_activations = sum(len(act_manager.get_activations(token)) for token in all_tokens)
    active_neurons = sum(1 for count in neuron_counts if count > 0)
    print(f"\nSummary:")
    print(f"Total activations processed: {total_activations}")
    print(f"Number of neurons activated at least once: {active_neurons}")
    print(f"Neuron with the highest activation count: {np.argmax(neuron_counts)} with {max(neuron_counts)} activations")

def main():
    analyze_neuron_activations()

if __name__ == "__main__":
    main()