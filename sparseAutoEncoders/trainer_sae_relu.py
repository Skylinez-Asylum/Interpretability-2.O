# Works well

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sae_relu import ReluAutoEncoder
from sae_dataset import SAE_Dataset
from torch.utils.data import Dataset, DataLoader
from print_color import print
config = {
    'activation_dim':768,
    'dict_dim':16384,
    'l1_coeff':3e-4,
    'batch_size': 128,
    'num_epochs': 1000,
    'lr':1e-4
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SAE_Dataset(10000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
print(train_size, val_size)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(dataset = train_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset = val_dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)

model = ReluAutoEncoder(cfg = config).to(device)
criterion = nn.MSELoss()
optimiser = optim.AdamW(model.parameters(), lr=config['lr'])

# training loop

for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    for x, _ in train_dataloader:
        x = x.to(device)
        optimiser.zero_grad()
        loss, x_reconstruct, acts, l2_loss, l1_loss = model.forward(x)  # loss is l1_loss + l2_loss
        print('Pure loss:', (l1_loss).item(), color='r')
        loss.backward()
        model.remove_parallel_component_of_grads()
        optimiser.step()
        # if epoch > config['num_epochs']: 
        #     optimiser = optim.AdamW(model.parameters(), lr=config['lr']/10)


    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = x.to(device)
            loss, _, _, _, _ = model.forward(x)
            val_loss += loss.item()
    val_loss /= len(test_dataloader)

    print(f'Epoch: {epoch+1} Loss: {loss} Validation Loss: {val_loss}')
