from model import load_model
from warnings import filterwarnings
import tiktoken
import torch
from torch.nn import functional as F
import numpy as np
from typing import List, Tuple, Dict
from activationManager import ActivationManager

filterwarnings('ignore')

class ActivationExtractor:
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
        self.model.to('cuda')
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.activation = None
        self.hook_handle = None
    
    def _activation_hook(self, module, input, output):
        self.activation = output.detach().cpu().numpy()
    
    def setup_hook(self):
        last_mlp = self.model.transformer.h[-1].mlp.c_proj
        self.hook_handle = last_mlp.register_forward_hook(self._activation_hook)
    
    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate_and_extract(
        self,
        prompt: str,
        max_length: int = 50,
        num_return_sequences: int = 1,
        batch_size=10,
    ) -> Tuple[List[str], List[np.ndarray], List[str]]:
        self.model.eval()
        self.setup_hook()
        
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
        # tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        tokens = tokens.unsqueeze(0).repeat(batch_size, 1)


        x = tokens
        
        all_activations = []
        all_tokens = []
        
        try:
            while x.size(1) < max_length:
                with torch.inference_mode():
                    logits, _ = self.model(x)
                    logits = logits[:, [-1], :]
                    probs = F.softmax(logits, dim=-1)
                    
                    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                    topk_probs = topk_probs.squeeze(1)
                    topk_indices = topk_indices.squeeze(1)
                    
                    ix = torch.multinomial(topk_probs, 1)
                    xcol = torch.gather(topk_indices, 1, ix)
                    
                    # Store activation for generated token
                    token_activation = self.activation[0, -1]
                    all_activations.append(token_activation)
                    
                    # Store token
                    token_text = self.tokenizer.decode([xcol[0].item()])
                    all_tokens.append(token_text)
                    
                    x = torch.cat((x, xcol), dim=1)
        finally:
            self.remove_hook()
        
        # Generate output texts
        generated_texts = []
        for i in range(num_return_sequences):
            tokens = x[i, :max_length].tolist()
            decode = self.tokenizer.decode(tokens)
            generated_texts.append(decode)
        
        return generated_texts, all_activations, all_tokens

def process_prompts_and_save_activations(
    prompts: List[str],
    activation_manager: ActivationManager,
    model_path: str,
    max_length: int = 100,
    num_sequences: int = 1
) -> Dict[str, List[str]]:
    extractor = ActivationExtractor(model_path)
    responses = {}
    
    for prompt in prompts:
        generated_texts, activations, tokens = extractor.generate_and_extract(
            prompt,
            max_length=max_length,
            num_return_sequences=num_sequences,
            batch_size=10
        )
        
        # Save activations for each token
        for activation, token in zip(activations, tokens):
            activation_manager.add_activation(activation, f"{token}")
        
        responses[prompt] = generated_texts
    
    return responses

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    path = r"customGPT2/save_states/state_step555000.pt"
    
    # Initialize activation manager
    manager = ActivationManager("activations/CustomGPT2/activations.pkl")
    
    # Test prompts
    prompts = [
"""Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question: Difference between RAM and ROM.
Answer:""", 

"""Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question:What is a banana?""",

'I am batman',
    ]
    
    # Process prompts and save activations
    responses = process_prompts_and_save_activations(
        prompts,
        manager,
        path,
        max_length=200,
        num_sequences=1
    )
    
    # Print generated responses
    # for prompt, texts in responses.items():
    #     print(f"\nPrompt: {prompt}")
    #     for i, text in enumerate(texts):
    #         print(f"Response {i+1}: {text}")
    
    # Display stats
    stats = manager.get_stats(show_vocabulary=True)
    print(f"\nStats after processing: {stats}")