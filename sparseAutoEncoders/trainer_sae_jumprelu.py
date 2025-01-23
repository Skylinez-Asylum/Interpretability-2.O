import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from sae_dataset import SAE_Dataset
from sae_jumprelu import JumpReluAutoEncoder

config = {
    'activation_dim': 768,
    'dict_dim': 16384,
    'l1_coeff': 3e-4,
    'batch_size': 128,
    'num_epochs': 200,
    'lr': 1e-4,
    'gradient_clip_val': 1.0,
    'checkpoint_frequency': 10,
    'weight_decay': 1e-5,
    'gradient_clip_val': 1.0,
    'checkpoint_steps': 1000,
    'resample_freq': 500,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = SAE_Dataset(5000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

# Model, optimizer, and scheduler
model = JumpReluAutoEncoder(cfg=config).to(device)
optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=config['lr']/100)

# Training history
history = {
    'train_loss': [],
    'val_loss': [],
    'train_l1_loss': [],
    'train_l2_loss': [],
}

def save_checkpoint(model, optimizer, epoch, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)


# Training loop

for epoch in range(config['num_epochs']):
    model.train()
    running_loss = 0.0
    running_l1_loss = 0.0
    running_l2_loss = 0.0
    
    for batch_idx, (x, _) in enumerate(train_dataloader):
        x = x.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        loss, x_reconstruct, acts, l2_loss, l1_loss = model(x)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_val'])
        
        # Update weights
        optimizer.step()
        scheduler.step()



        # Update running losses
        running_loss += loss.item()
        running_l1_loss += l1_loss.item()
        running_l2_loss += l2_loss.item()
    
    # Calculate average training losses
    avg_train_loss = running_loss / len(train_dataloader)
    avg_l1_loss = running_l1_loss / len(train_dataloader)
    avg_l2_loss = running_l2_loss / len(train_dataloader)
    
    # Validation phase
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for x, _ in val_dataloader:
            x = x.to(device)
            loss, _, _, _, _ = model(x)
            val_running_loss += loss.item()
    
    avg_val_loss = val_running_loss / len(val_dataloader)
    
    # Update history
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['train_l1_loss'].append(avg_l1_loss)
    history['train_l2_loss'].append(avg_l2_loss)
    
    print(f'Epoch: {epoch+1}/{config["num_epochs"]} ||| Train Loss: {avg_train_loss:.6f} L1: {avg_l1_loss:.6f} L2: {avg_l2_loss:.6f} ||| Val Loss: {avg_val_loss:.6f}')

    # Save checkpoint
    # if (epoch + 1) % config['checkpoint_frequency'] == 0:
    #     save_checkpoint(model, optimizer, epoch, avg_train_loss, 
    #                    f'checkpoint_epoch_{epoch+1}.pt')


# Plotting function
def run_plot():
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # L1 vs L2 loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history['train_l1_loss'], label='L1 Loss')
    plt.plot(history['train_l2_loss'], label='L2 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Component')
    plt.title('L1 vs L2 Loss Components')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Run the plot
# run_plot()