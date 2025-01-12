import torch.nn.functional as F
import torch.nn as nn
import torch
from dataclasses import dataclass
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Union, Optional
# np.set_printoptions(threshold=np.inf) # to print numpy array properly


@dataclass
class GPTConfig:
        block_size:int = 1024
        vocab_size:int = 50304
        n_layer:int   = 12
        n_head:int = 6
        n_embd:int = 768

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x, layer_ct):
        x = self.c_fc(x)
        x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        if layer_ct == 11: # last layer
            # print(x.shape)
            # saving activations
            processed_activations = x.detach().cpu().numpy()
            # np.save('customGPT-2/save_states/activations.npy', np.array(processed_activations, dtype=object))
            # print(processed_activations.shape)
            # print(list(processed_activations))
            # print(processed_activations)
            # quit()
            print(processed_activations.shape)

            activation_list.append(processed_activations)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, layer_ct):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)), layer_ct)
        return x

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

    def forward(self, idx, targets=None, return_logits=True):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        for layer_ct, block in enumerate(self.transformer.h):
            # print(layer_ct)
            x = block(x, layer_ct)
        x = F.rms_norm(x, (x.size(-1),))

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            logits = logits.float() # use tf32/fp32 for logits
            loss = None
        # there are performance reasons why not returning logits is prudent, if not needed
        if not return_logits:
            logits = None

        return logits, loss



# Data structure

class WordActivationStorage:
    """
    Stores and retrieves multiple activations per word using HDF5.
    """
    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        self.initialize_storage()
    
    def initialize_storage(self):
        """Create or open the HDF5 file."""
        with h5py.File(self.filepath, 'a') as f:
            if 'word_activations' not in f:
                f.create_group('word_activations')
    
    def add_activation(self, word: str, activation: np.ndarray):
        """
        Add a new activation for a word.
        
        Args:
            word: The word/token
            activation: numpy array of activation values
        """
        with h5py.File(self.filepath, 'a') as f:
            # Create word group if it doesn't exist
            if word not in f['word_activations']:
                word_group = f['word_activations'].create_group(word)
            else:
                word_group = f['word_activations'][word]
            
            # Create new dataset with incremental index
            next_idx = len(word_group.keys())
            word_group.create_dataset(
                str(next_idx),
                data=activation,
                compression='gzip'
            )
    
    def add_activations_batch(self, word: str, activations: np.ndarray):
        """
        Add multiple activations for a word at once.
        
        Args:
            word: The word/token
            activations: numpy array of shape (n_instances, activation_dim)
        """
        with h5py.File(self.filepath, 'a') as f:
            if word not in f['word_activations']:
                word_group = f['word_activations'].create_group(word)
            else:
                word_group = f['word_activations'][word]
            
            start_idx = len(word_group.keys())
            for i, activation in enumerate(activations):
                word_group.create_dataset(
                    str(start_idx + i),
                    data=activation,
                    compression='gzip'
                )
    
    def get_all_activations(self, word: str) -> np.ndarray:
        """
        Get all stored activations for a word.
        
        Returns:
            numpy array of shape (n_instances, activation_dim)
        """
        with h5py.File(self.filepath, 'r') as f:
            if word not in f['word_activations']:
                return np.array([])
            
            word_group = f['word_activations'][word]
            # Sort by index to maintain order
            indices = sorted([int(k) for k in word_group.keys()])
            return np.array([word_group[str(i)][:] for i in indices])
    
    def get_activation(self, word: str, index: int) -> Optional[np.ndarray]:
        """Get a specific activation instance for a word."""
        with h5py.File(self.filepath, 'r') as f:
            if word not in f['word_activations']:
                return None
            
            word_group = f['word_activations'][word]
            if str(index) not in word_group:
                return None
                
            return word_group[str(index)][:]
    
    def get_activation_count(self, word: str) -> int:
        """Get the number of stored activations for a word."""
        with h5py.File(self.filepath, 'r') as f:
            if word not in f['word_activations']:
                return 0
            return len(f['word_activations'][word].keys())
    
    def list_words(self) -> List[str]:
        """Get list of all stored words."""
        with h5py.File(self.filepath, 'r') as f:
            return list(f['word_activations'].keys())
    
    def get_statistics(self, word: str) -> Dict:
        """Get statistics about stored activations for a word."""
        activations = self.get_all_activations(word)
        if len(activations) == 0:
            return {"count": 0}
        
        return {
            "count": len(activations),
            "mean": np.mean(activations, axis=0),
            "std": np.std(activations, axis=0),
            "min": np.min(activations, axis=0),
            "max": np.max(activations, axis=0)
        }

def load_single_word(filepath: str, word: str) -> Optional[np.ndarray]:
    """
    Load activations for a specific word, handling variable sequence lengths.
    Returns a list of numpy arrays, each representing one instance of activation.
    """
    with h5py.File(filepath, 'r') as f:
        if word not in f['word_activations']:
            return None
        
        word_group = f['word_activations'][word]
        indices = sorted([int(k) for k in word_group.keys()])
        
        # Create list of activations without trying to stack them
        activations = []
        for i in indices:
            activation = word_group[str(i)][:]
            activations.append(activation)
            
        return activations

def load_activations(filepath: str) -> Dict[str, List[np.ndarray]]:
    """
    Load all word activations from an HDF5 file.
    Returns a dictionary mapping words to lists of activation arrays.
    """
    data = {}
    with h5py.File(filepath, 'r') as f:
        words = list(f['word_activations'].keys())
        
        for word in words:
            word_group = f['word_activations'][word]
            indices = sorted([int(k) for k in word_group.keys()])
            
            # Create list of activations for this word
            activations = []
            for i in indices:
                activation = word_group[str(i)][:]
                activations.append(activation)
                
            data[word] = activations
            
    return data

def print_activation_info(filepath: str, word: str):
    """
    Print detailed information about activations for a specific word.
    """
    with h5py.File(filepath, 'r') as f:
        if word not in f['word_activations']:
            print(f"Word '{word}' not found in file")
            return
            
        word_group = f['word_activations'][word]
        indices = sorted([int(k) for k in word_group.keys()])
        
        print(f"\nActivation info for word '{word}':")
        print(f"Number of instances: {len(indices)}")
        print("\nShapes for each instance:")
        for i in indices:
            activation = word_group[str(i)][:]
            print(f"Instance {i}: {activation.shape}")


# basic functoins
def remove_prefix(state_dict, prefix):
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

def load_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cuda', weights_only=True)
    model_state_dict = checkpoint['model']
    model_state_dict = remove_prefix(model_state_dict, "_orig_mod.")
    new_model = GPT(GPTConfig)  
    new_model.load_state_dict(model_state_dict)
    return new_model

def extract_activations(model, inp):
    _, tokens = inference(model,inp,30,1, extracting_activations=True)
    print(len(tokens), len(activation_list))
    assert len(tokens) - 1 == len(activation_list), 'A bug has to be fixed which wont allow initial tokens to be greater than 1' # activation for the last token will not be generated 

    for token, activation in zip(tokens, activation_list):

        storage.add_activation(str(token), activation)








if __name__ == '__main__':

    # from inference import inference
    # import pandas as pd
    # import tiktoken

    # enc = tiktoken.get_encoding('gpt2')
    # path = r"customGPT-2/save_states/FT50k.pt"
    # storage = WordActivationStorage('word_activations.h5')

    # print('Loading model...')
    # model = load_model(path).to('cuda')

    # activation_dict = {}
    # for i in ['hey', 'why', 'How', 'a', ' ', 'hello', 'cow', 'talk', 'My', 'name']: # ensure activation starts with only one token

    #     activation_list = []

    #     extract_activations(model, i)

    # # print(activation_dict)

    # df = pd.DataFrame.from_dict(activation_dict, orient="index")
    # df.to_csv("activations.csv", index=True)   

    
    '''Load single word activations '''

    activations = load_single_word('word_activations.h5', '12')

    # Print information about each activation
    print("\nActivation lengths:")
    for i, act in enumerate(activations):
        print(f"Instance {i} shape: {act.shape}")
        print(act)


    '''Load all activations'''
    # all_data = load_activations('word_activations.h5')