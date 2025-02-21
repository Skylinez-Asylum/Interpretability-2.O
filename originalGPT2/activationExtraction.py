from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from typing import List, Tuple, Dict
from activationManager import ActivationManager

class ActivationExtractor:
    def __init__(self, model_name: str = "Arjun-G-Ravi/chat-GPT2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.activation = None
        self.hook_handle = None
        
    def _activation_hook(self, module, input, output):
        # Store the full sequence of activations
        self.activation = output.detach().cpu().numpy()
    
    def setup_hook(self):
        # Find the last MLP layer in the model
        last_block = self.model.transformer.h[-1]
        mlp = last_block.mlp.c_proj  # The final projection layer of the MLP
        
        # Register the forward hook
        self.hook_handle = mlp.register_forward_hook(self._activation_hook)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_and_extract(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_length: int = 200
    ) -> Tuple[str, List[np.ndarray], List[str]]:
        self.setup_hook()
        all_activations = []
        all_tokens = []
        
        # Tokenize the input
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True
        )
        
        # Generate text token by token
        input_ids = encoding.input_ids
        attention_mask = encoding.attention_mask
        current_length = input_ids.shape[1]
        
        with torch.no_grad():
            while current_length < max_length:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Get probabilities for next token
                next_token_logits = outputs.logits[0, -1, :]
                if temperature > 0:
                    probs = torch.nn.functional.softmax(next_token_logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits).unsqueeze(0)
                
                # Store activation for the generated token
                token_activation = self.activation[0, -1]  # Get activation for the last position
                all_activations.append(token_activation)
                
                # Store the token
                token_text = self.tokenizer.decode(next_token)
                all_tokens.append(token_text)
                
                # Update input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1)
                current_length += 1
                
                # Check for EOS token
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        self.remove_hook()
        return generated_text, all_activations, all_tokens

def process_prompts_and_save_activations(
    prompts: List[str],
    activation_manager: ActivationManager,
    category: str = None,
    model_name: str = "Arjun-G-Ravi/chat-GPT2",
    temperature: float = 0.5,
    max_length: int = 100
) -> Dict[str, str]:
    """
    Process a list of prompts, generate responses, and save token-by-token activations.
    """
    extractor = ActivationExtractor(model_name)
    responses = {}
    
    for i, prompt in enumerate(prompts):
        print(f'{i}/{len(prompts)}: {prompt}')
        # Generate text and extract activations for each token
        generated_text, activations, tokens = extractor.generate_and_extract(
            prompt,
            temperature=temperature,
            max_length=max_length
        )
        
        # Save activations for each token
        for activation, token in zip(activations, tokens):
            activation_manager.add_activation(activation, f"{token}")
        
        # Store the response    
        responses[prompt] = generated_text
    
    return responses

if __name__ == "__main__":
    # Example usage
    manager = ActivationManager("activations/GPT2FT/activations.pkl")
    print(torch.cuda.is_available())
    
    with open('activationDataset.txt', 'r') as f:
        text = f.read()
    if not text:
        print('text not found')
        quit()
 
    prompts = list(text.split('\n'))[:25]
    
    # Process the prompts and save their activations
    responses = process_prompts_and_save_activations(
        prompts,
        manager,
    )
    
    # Print the generated responses
    # for prompt, response in responses.items():
    #     print(f"\nPrompt: {prompt}\nResponse: {response}\n")
        
    # Display stats
    stats = manager.get_stats(show_vocabulary=True)
    print(f"Stats after processing: {stats}")