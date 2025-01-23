import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from sae_dataset import SAE_Dataset
from torch.utils.data import Dataset, DataLoader
from sae_topk import TopKAutoEncoder
def train_topk_sae(config):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset preparation
    dataset = SAE_Dataset(5000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # DataLoaders
    train_dataloader = DataLoader(
        dataset=train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=0
    )
    
    # Model initialization
    model = TopKAutoEncoder(cfg=config).to(device)
    
    # Optimizer with adaptive learning rate
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=10, 
        min_lr=1e-5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_l2_loss': [],
        'train_l1_loss': [],
        'lr': []
    }
    
    # Training loop
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        running_l2_loss = 0.0
        running_l1_loss = 0.0
        
        for batch_idx, (x, _) in enumerate(train_dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            loss, x_reconstruct, acts, l2_loss, l1_loss = model(x)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('gradient_clip_val', 1.0))
            
            # Optimizer step
            optimizer.step()
            
            running_loss += loss.item()
            running_l2_loss += l2_loss.item()
            running_l1_loss += l1_loss.item()
        
        # Calculate average training losses
        avg_train_loss = running_loss / len(train_dataloader)
        avg_l2_loss = running_l2_loss / len(train_dataloader)
        avg_l1_loss = running_l1_loss / len(train_dataloader)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        
        with torch.no_grad():
            for x, _ in val_dataloader:
                x = x.to(device)
                loss, _, _, _, _ = model(x)
                val_running_loss += loss.item()
        
        avg_val_loss = val_running_loss / len(val_dataloader)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_l2_loss'].append(avg_l2_loss)
        history['train_l1_loss'].append(avg_l1_loss)
        history['lr'].append(current_lr)
        
        # Print epoch statistics
        print(f'Epoch: {epoch+1}/{config["num_epochs"]} '
              f'||| Train Loss: {avg_train_loss:.6f} '
              f'L2: {avg_l2_loss:.6f} '
              f'L1: {avg_l1_loss:.6f} '
              f'||| Val Loss: {avg_val_loss:.6f} '
              f'||| LR: {current_lr:.6f}')
    
    return model, history

def plot_training_history(history):
    plt.figure(figsize=(15, 10))
    
    # Loss plots
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # L2 Loss plot
    plt.subplot(2, 2, 2)
    plt.plot(history['train_l2_loss'], label='Training L2 Loss')
    plt.title('L2 Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L2 Loss')
    plt.legend()
    
    # L1 Loss plot
    plt.subplot(2, 2, 3)
    plt.plot(history['train_l1_loss'], label='Training L1 Loss')
    plt.title('L1 Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.legend()
    
    # Learning Rate plot
    plt.subplot(2, 2, 4)
    plt.plot(history['lr'], label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Configuration
config = {
    'activation_dim': 768,
    'dict_dim': 16384,
    'l1_coeff': 3e-4,
    'batch_size': 32,
    'num_epochs': 200,
    'lr': 5e-4,
    'k': 5,
    'gradient_clip_val': 1.0,
    'weight_decay': 1e-5
}

# Training
model, history = train_topk_sae(config)

# Optionally plot training history
# plot_training_history(history)