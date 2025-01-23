import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import tiktoken
from datasets import load_dataset

# Configuration
local_dir = "flanv2_preprocessed"
shard_size = int(1e8)  
DATA_CACHE_DIR = os.path.join(os.getcwd(), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

dataset = load_dataset("chiayewken/flan-v2")

# Initialize the GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # End of text token

def tokenize_instance(instance):
    """
    Tokenize a single instance using GPT-2 tokenizer.
    """
    instruction = instance["source"]
    response = instance["target"]
    
    # Concatenate source and target with an EOT token
    text = f"{instruction}<|endoftext|>{response}<|endoftext|>"
    tokens = enc.encode_ordinary(text)
    tokens.append(eot)  # Append the final EOT token
    tokens_np = np.array(tokens, dtype=np.uint16)
    
    # Ensure tokens fit within uint16
    assert (tokens_np >= 0).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    return tokens_np

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def preprocess_and_save(dataset, split_name):
    nprocs = max(1, os.cpu_count() // 2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)  # Preallocate buffer
        token_count = 0
        progress_bar = None

        for tokens in pool.imap(tokenize_instance, dataset, chunksize=16):
            if token_count + len(tokens) < shard_size:
                # Append tokens to current shard
                all_tokens_np[token_count:token_count + len(tokens)] = tokens
                token_count += len(tokens)

                # Initialize or update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"{split_name} Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # Write current shard and start a new one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_shard_{shard_index:06d}.npy")
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None

                # Populate the next shard with leftover tokens
                all_tokens_np[:len(tokens) - remainder] = tokens[remainder:]
                token_count = len(tokens) - remainder

        # Write any remaining tokens as the last shard
        if token_count != 0:
            filename = os.path.join(DATA_CACHE_DIR, f"{split_name}_shard_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])

# Preprocess and save each split
for split_name, split_data in dataset.items():
    preprocess_and_save(split_data, split_name)

print(f"Preprocessing complete. Shards saved to {DATA_CACHE_DIR}")
