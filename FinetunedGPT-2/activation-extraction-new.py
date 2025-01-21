from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from collections import defaultdict
import numpy as np
import json
import os
from datetime import datetime

class ActivationExtractor:
    def __init__(self, model_name="Arjun-G-Ravi/chat-GPT2", save_dir="activations"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.activations = None
        self.word_activations = defaultdict(list)
        self.save_dir = save_dir
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Register hook for the last MLP layer
        for name, module in reversed(list(self.model.named_modules())):
            if 'mlp' in name.lower():
                module.register_forward_hook(self._get_activation())
                break
    
    def _get_activation(self):
        def hook(module, input, output):
            self.activations = output.detach()
        return hook
    
    def generate_and_extract(self, prompt, temperature=0.7, max_length=200):
        # Tokenize and generate as before
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_return_sequences=1,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                output_hidden_states=True
            )
        
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Process activations and map them to words
        words = self.tokenizer.convert_ids_to_tokens(output_ids[0])
        for idx, word in enumerate(words):
            if idx < self.activations.shape[1]:
                self.word_activations[word].append(self.activations[0, idx, :].numpy())
        
        # Save activations to file
        self._save_activations(prompt, generated_text)
        
        return generated_text, self.word_activations
    
    def _save_activations(self, prompt, generated_text):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activations_{timestamp}"
        
        # Save numpy arrays
        np_filename = os.path.join(self.save_dir, f"{filename}.npz")
        activation_dict = {
            word: np.array(activations) 
            for word, activations in self.word_activations.items()
        }
        np.savez_compressed(np_filename, **activation_dict)
        
        # Save metadata
        meta_filename = os.path.join(self.save_dir, f"{filename}_meta.json")
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "generated_text": generated_text,
            "activation_file": np_filename,
            "word_counts": {
                word: len(activations) 
                for word, activations in self.word_activations.items()
            }
        }
        with open(meta_filename, 'w') as f:
            json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    extractor = ActivationExtractor()
    prompt = """Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question: Difference between RAM and ROM.
Answer:"""
    
    generated_text, word_activations = extractor.generate_and_extract(prompt)
    print("Generated text:", generated_text)
    print("\nActivations saved to 'activations' directory")