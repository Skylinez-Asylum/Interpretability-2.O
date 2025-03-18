'''
You can give a token and this code will run its activation through the SAE of your choice and will give the neurons that activated for each of those tokens.
'''
import torch
import numpy as np
from sae_jumprelu import JumpReluAutoEncoder
from activationManager import ActivationManager

def analyze_token_activations(token: str, 
                            checkpoint_path: str = '/home/arjun/Desktop/GitHub/Interpretability-2.O/sparseAutoEncoders/save_states/CustomFT_jumprelu/model_v12_200.pt',
                            activations_storage: str = '/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations.pkl',
                            batch_size: int = 32,
                            normalization_factor: float = 1000.0):
    """
    Analyze activations for a given token through the SAE and report activated neurons.
    Activations are normalized by dividing by normalization_factor.
    
    Args:
        token (str): The token to analyze (e.g., 'cat', 'dog')
        checkpoint_path (str): Path to the trained SAE model checkpoint
        activations_storage (str): Path to the stored activations
        batch_size (int): Batch size for processing activations
        normalization_factor (float): Factor to divide activations by for normalization
    """
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    config = {
        'activation_dim': 768,
        'dict_dim': 16384,
        'l1_coeff': 3e-4,
    }
    
    model = JumpReluAutoEncoder(cfg=config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}, epoch {checkpoint['epoch']}")
    
    # Load activations
    act_manager = ActivationManager(storage_path=activations_storage)
    token_activations = act_manager.get_activations(token)
    
    if not token_activations:
        print(f"No activations found for token '{token}'")
        return
    
    print(f"Found {len(token_activations)} activations for token '{token}'")
    
    # Convert to tensor and normalize
    activations_array = np.array(token_activations)
    normalized_activations = activations_array / normalization_factor
    activations_tensor = torch.tensor(normalized_activations).to(device)
    
    # Process activations in batches
    activated_neurons_info = []
    
    with torch.no_grad():
        for batch_start in range(0, len(token_activations), batch_size):
            batch_end = min(batch_start + batch_size, len(token_activations))
            batch = activations_tensor[batch_start:batch_end]
            
            # Run through SAE
            _, _, acts, _, _ = model(batch)
            
            # Find activated neurons (where activation > 0)
            active_mask = acts > 0
            for i in range(active_mask.shape[0]):  # For each activation in batch
                active_indices = torch.where(active_mask[i])[0].cpu().numpy()
                num_active = len(active_indices)
                activated_neurons_info.append({
                    'activation_idx': batch_start + i,
                    'num_active': num_active,
                    'active_indices': active_indices
                })
            
            # Clear memory
            del batch
            torch.cuda.empty_cache()
    
    # Print results
    print(f"\nAnalysis for token '{token}' (activations normalized by {normalization_factor}):")
    print(f"Total activations processed: {len(token_activations)}")
    print("\nPer-activation breakdown:")
    for info in activated_neurons_info:
        print(f"Activation {info['activation_idx']}:")
        print(f"  Number of activated neurons: {info['num_active']}")
        print(f"  Indices of activated neurons: {info['active_indices'].tolist()}")
    
    # Summary statistics
    num_active_list = [info['num_active'] for info in activated_neurons_info]
    if num_active_list:
        print("\nSummary:")
        print(f"Average number of active neurons: {np.mean(num_active_list):.2f}")
        print(f"Minimum number of active neurons: {min(num_active_list)}")
        print(f"Maximum number of active neurons: {max(num_active_list)}")
    
    # Find most common neurons
    all_active_indices = np.concatenate([info['active_indices'] for info in activated_neurons_info])
    unique_indices, counts = np.unique(all_active_indices, return_counts=True)
    print(f'\nToken:{token}')
    print(f'Total activations: {len(num_active_list)}')
    print("\nMost frequently activated neurons:")
    for idx, count in zip(unique_indices[:5], counts[:5]):  # Top 5
        print(f"Neuron {idx}: activated {count} times")

def main():
    override = False
    while True and override:
        token = input("Enter a token to analyze (or 'quit' to exit): ").strip()
        if token.lower() == 'quit':
            break
        analyze_token_activations(token)
        print("\n" + "="*50 + "\n")
    
    if not override:
        token = ' cow'
        analyze_token_activations(token)

if __name__ == "__main__":
    main()