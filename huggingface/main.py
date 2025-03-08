from transformers import PreTrainedModel, PretrainedConfig
import torch
from dataclasses import dataclass
from model import Block  # Assuming this is correctly imported
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

# Define a custom configuration class compatible with Hugging Face
class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"  # Unique identifier for your model
    def __init__(self, block_size=1024, vocab_size=50304, n_layer=12, n_head=6, n_embd=768, **kwargs):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

# Adapt your GPT class to inherit from PreTrainedModel
class GPT(PreTrainedModel):
    config_class = CustomGPTConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True):
        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = F.rms_norm(x, (x.size(-1),))
        if targets is not None:
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits.float()
            loss = None
        if not return_logits:
            logits = None
        return logits, loss

# Save your model
def save_model_to_hf_format(model, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the configuration
    config = CustomGPTConfig(
        block_size=model.config.block_size,
        vocab_size=model.config.vocab_size,
        n_layer=model.config.n_layer,
        n_head=model.config.n_head,
        n_embd=model.config.n_embd
    )
    config.save_pretrained(output_dir)
    
    # Save the model weights with safe_serialization=False
    model.save_pretrained(output_dir, safe_serialization=False)

# Example usage
if __name__ == "__main__":
    config = CustomGPTConfig()  # Use CustomGPTConfig instead of GPTConfig
    model = GPT(config)
    save_model_to_hf_format(model, "custom_gpt_model")