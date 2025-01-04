from dataclasses import dataclass

@dataclass
class GPTConfig:
        block_size:int = 1024
        vocab_size:int = 50304
        n_layer:int   = 12
        n_head:int = 6
        n_embd:int = 768