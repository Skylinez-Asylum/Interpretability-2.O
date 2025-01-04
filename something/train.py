import torch
import torch.nn as nn
import tiktoken
from model import GPT
from config import GPTConfig
from dataloader import DataLoaderLite

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")


model = GPT(GPTConfig())

model.to(device)
train_loader = DataLoaderLite(B=4, T=32)

optimizer = model.configure_optimizers(weight_decay=0.1,
                                               learning_rate=1e-4, betas=(0.9, 0.95))
for i in range(50):
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()
    optimizer.step()
    print(f"step {i}, loss: {loss.item()}")
