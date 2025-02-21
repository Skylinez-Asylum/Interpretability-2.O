import torch
import numpy as np
from tqdm import trange
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Tuple, Dict
from activationManager import ActivationManager


class ActivationExtractor:
    def __init__(self, model_name: str = "Arjun-G-Ravi/chat-GPT2", device: str = "cuda", batch_size: int = 8):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device
        self.batch_size = batch_size
        
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        self.activation = None
        self.hook_handle = None
    
    def _activation_hook(self, module, input, output):
        self.activation = output.detach().cpu().numpy()
    
    def setup_hook(self):
        last_block = self.model.module.transformer.h[-1] if isinstance(self.model, torch.nn.DataParallel) else self.model.transformer.h[-1]
        mlp = last_block.mlp.c_proj  # Final projection layer of MLP
        self.hook_handle = mlp.register_forward_hook(self._activation_hook)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
    
    def generate_and_extract(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_length: int = 200
        ):
        '''This is the main function. This one generates text from the model, and returns activations. This only takes activation of newly generated tokens.'''

        self.setup_hook()
        encoding = self.tokenizer( prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=True).to(self.device)  
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        batch_size = input_ids.shape[0]
        
        generated_texts = ["" for _ in range(batch_size)]
        batch_activations = [[] for _ in range(batch_size)]
        batch_tokens = [[] for _ in range(batch_size)]
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]
                
                if temperature > 0:
                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                for i in range(batch_size):
                    token_activation = self.activation[i, -1]  
                    batch_activations[i].append(token_activation)
                    token_text = self.tokenizer.decode(next_tokens[i])
                    batch_tokens[i].append(token_text)
                    # generated_texts[i] += token_text # commented to save time
                
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=self.device)], dim=1)
                
                if torch.any(next_tokens.squeeze() == self.tokenizer.eos_token_id):
                    break
        
        self.remove_hook()
        return generated_texts, batch_activations, batch_tokens


def process_prompts_and_save_activations(
    prompts: List[str],
    activation_manager: ActivationManager,
    model_name: str = "Arjun-G-Ravi/chat-GPT2",
    temperature: float = 0.5,
    max_length: int = 200,
    batch_size: int = 24 # default
) -> Dict[str, str]:
    extractor = ActivationExtractor(model_name, batch_size=batch_size)
    responses = {}

    for i in trange(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        generated_texts, activations, tokens = extractor.generate_and_extract(
            batch_prompts,
            temperature=temperature,
            max_length=max_length
        )
        
        for j, prompt in enumerate(batch_prompts):
            for activation, token in zip(activations[j], tokens[j]):
                activation_manager.add_activation(activation, token)
                activation_manager._save_activations()
            responses[prompt] = generated_texts[j]
    
    return responses


if __name__ == "__main__":
    manager = ActivationManager("activations/GPT2FT/activations.pkl")
    print('GPU available:', torch.cuda.is_available())
    
    with open('activationDataset.txt', 'r') as f:
        text = f.read()
    if not text:
        print('text not found')
        quit()
    
    prompts = text.split('\n')[:10]
    
    responses = process_prompts_and_save_activations(
        prompts,
        manager,
        batch_size=24
    )
    
    stats = manager.get_stats(show_vocabulary=True)
    print(f"Stats after processing: {stats}")
