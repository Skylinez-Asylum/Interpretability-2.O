
# some bug, the loss doesnt go down

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
from sae_topk import TopKAutoEncoder, config
from sae_dataset import SAE_Dataset
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = SAE_Dataset()
train_dataloader = DataLoader(dataset = dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)
test_dataloader = DataLoader(dataset = dataset, batch_size = config['batch_size'], shuffle=True, num_workers=0)

model = TopKAutoEncoder(cfg = config).to(device)
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=config['lr'])

# training loop
for epoch in range(config['num_epochs']):
    model.train()
    total_loss = 0
    for x,_ in train_dataloader:
        x = x.to(device)
        optimiser.zero_grad()
        # print('cow 1')
        loss, x_reconstruct, acts, l2_loss, l1_loss =  model.forward(x)  # loss is l1_loss + l2_loss
        # print('cow 2')
        loss.backward()
        optimiser.step()
    print(f'Epoch: {epoch+1} Loss: {loss}')

        
