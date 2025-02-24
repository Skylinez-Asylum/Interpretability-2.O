from sparseAutoEncoders.activationManager import ActivationManager

manager = ActivationManager("/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2FT/activations.pkl")


stats = manager.get_stats(show_vocabulary=True)
print(f"Activation stats: {stats}")


print('---')

act = manager.get_activations(' is')
print(len(act))
print(act[0].shape)
