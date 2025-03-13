from GPT2FT.activationManager import ActivationManager
import numpy as np

# To convert
manager = ActivationManager("/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/CustomGPT2FT/activations.pkl")
print(f"Activation stats: {manager.get_stats()}")
activations = []
for k,v in manager._load_activations().items():
    activations.extend(v)

print(len(activations))
activations = np.array(activations)
np.save('activations/CustomGPT2FT/activations.npy', activations)

# # To load data
# data = np.load('GPT2activations.npy')
# print(data.shape)