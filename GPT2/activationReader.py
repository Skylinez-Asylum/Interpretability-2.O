from activationManager import ActivationManager

manager = ActivationManager("/home/arjun/Desktop/GitHub/Interpretability-2.O/activations/GPT2/activationsJJ.pkl")


stats = manager.get_stats(show_vocabulary=False)
print(f"Activation stats: {stats}")


print('---')

act = manager.get_activations(' is')
print(len(act))
print(act[0].shape)
# print(act[1000])