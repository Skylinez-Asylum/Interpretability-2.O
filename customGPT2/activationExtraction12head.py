from activationManager import ActivationManager
from model import load_model
from warnings import filterwarnings
import tiktoken
import torch
from torch.nn import functional as F
import numpy as np
from typing import List, Dict
from pathlib import Path
import pickle
from tqdm import trange

filterwarnings('ignore')

class ActivationExtractor:
    def __init__(self, model_path: str, batch_size: int = 64, attention_head=6):  # Increased default batch size
        if attention_head!=6:
            from dataclasses import dataclass

            @dataclass
            class GPTConfig:
                    block_size:int = 1024
                    vocab_size:int = 50304
                    n_layer:int   = 12
                    n_head:int = 12
                    n_embd:int = 768
            self.model = load_model(path, GPTConfig)
        else: self.model = load_model(model_path)
        self.model.to('cuda')
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.batch_size = batch_size
        self.activation = None
        self.hook_handle = None
    
    def _activation_hook(self, module, input, output):
        self.activation = output[:, -1, :].detach().cpu().numpy()
    
    def setup_hook(self):
        last_mlp = self.model.transformer.h[-1].mlp.c_proj
        self.hook_handle = last_mlp.register_forward_hook(self._activation_hook)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_and_extract(
        self,
        prompts: List[str],
        max_length: int = 50,
        temperature: float = 0.5,
    ):
        self.model.eval()
        self.setup_hook()
        
        # Tokenize and prepare batch
        tokens = [self.tokenizer.encode(prompt) for prompt in prompts]
        max_prompt_len = max(len(t) for t in tokens)
        tokens = [t + [self.tokenizer.eot_token] * (max_prompt_len - len(t)) for t in tokens]
        tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
        
        # Pre-allocate tensor for max_length to maximize VRAM usage early
        batch_size = tokens.size(0)
        x = torch.zeros((batch_size, max_length), dtype=torch.long, device='cuda')
        x[:, :tokens.size(1)] = tokens
        
        all_activations = [[] for _ in range(batch_size)]
        all_tokens = [[] for _ in range(batch_size)]
        
        try:
            with torch.inference_mode():
                for step in range(max_length - tokens.size(1)):
                    logits, _ = self.model(x[:, :tokens.size(1) + step])  # Process up to current length
                    logits = logits[:, -1, :]
                    probs = F.softmax(logits / temperature, dim=-1)
                    
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, 1, ix)
                    
                    for i in range(batch_size):
                        token_activation = self.activation[i]
                        all_activations[i].append(token_activation)
                        token_text = self.tokenizer.decode([xcol[i].item()])
                        all_tokens[i].append(token_text)
                    
                    x[:, tokens.size(1) + step] = xcol.squeeze(1)
        finally:
            self.remove_hook()
        
        # Generate output texts
        generated_texts = []
        for i in range(batch_size):
            tokens = x[i, :max_length].tolist()
            decode = self.tokenizer.decode([t for t in tokens if t != self.tokenizer.eot_token])  # Remove padding
            generated_texts.append(decode)
        
        return generated_texts, all_activations, all_tokens

def process_prompts_and_save_activations(
    prompts: List[str],
    activation_manager: ActivationManager,
    model_path: str,
    max_length: int = 200,
    temperature: float = 0.5,
    num_sequences: int = 1,
    attention_head=6,
    batch_size: int = 64  # Increased default batch size
) -> Dict[str, List[str]]:
    extractor = ActivationExtractor(model_path, batch_size=batch_size, attention_head=attention_head)
    responses = {}
    
    for i in trange(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        generated_texts, activations, tokens = extractor.generate_and_extract(
            batch_prompts,
            max_length=max_length,
            temperature=temperature,
        )
        
        for j, prompt in enumerate(batch_prompts):
            for activation, token in zip(activations[j], tokens[j]):
                activation_manager.add_activation(activation, token)
            responses[prompt] = [generated_texts[j]]
    
    activation_manager._save_activations()
    
    return responses

class ActivationManager:
    def __init__(self, storage_path: str = 'activations/CustomGPT2/activations.pkl'):
        self.storage_path = Path(storage_path)
        self.activations: Dict[str, List[np.ndarray]] = self._load_activations()
    
    def _load_activations(self) -> Dict[str, List[np.ndarray]]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    loaded_data = pickle.load(f)
                    return {category: list(acts) if not isinstance(acts, list) else acts 
                            for category, acts in loaded_data.items()}
            except Exception as e:
                print(f"Error loading activations: {e}")
                return {}
        return {}
    
    def _save_activations(self) -> None:
        with open(self.storage_path, 'wb') as f:
            pickle.dump(self.activations, f)
    
    def add_activation(self, activation: np.ndarray, category: str) -> None:
        if category not in self.activations:
            self.activations[category] = []
        self.activations[category].append(activation)
    
    def get_activations(self, category: str) -> List[np.ndarray]:
        return self.activations.get(category, [])
    
    def get_stats(self, show_vocabulary: bool = False) -> Dict[str, int]:
        total_count = sum(len(acts) for acts in self.activations.values())
        if show_vocabulary:
            return {"total": total_count, "Vocabulary": {k: len(v) for k, v in self.activations.items()}}
        return {"total": total_count}

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    path = r"customGPT2/save_states/12headNFT900k.pt"
    
    manager = ActivationManager("activations/CustomGPT2/activations.pkl")
    print('GPU available:', torch.cuda.is_available())
    
    with open('activationDataset.txt', 'r') as f:
        text = f.read()
    if not text:
        print('text not found')
        quit()
    prompts = text.split('\n')
    
    responses = process_prompts_and_save_activations(
        prompts,
        manager,
        path,
        max_length=200,
        temperature=0.5,
        num_sequences=1,
        batch_size=512,  # Increased to use more VRAM,
        attention_head=12
    )
    
    stats = manager.get_stats(show_vocabulary=True)
    print(f"\nStats after processing: {stats}")