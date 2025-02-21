import torch
from transformers import AutoModel, AutoConfig, AutoTokenizer
import tiktoken

class CustomModelInterface:
    def __init__(self, model_name="Arjun-G-Ravi/Custom-GPT-555k"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = tiktoken.get_encoding('gpt2')
        
    def generate(self, prompt, max_length=50, num_return_sequences=1, temperature=1.0):
        # Encode the input text
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        input_ids = input_ids.unsqueeze(0).repeat(num_return_sequences, 1)
        
        # Generate output tokens
        with torch.inference_mode():
            for _ in range(max_length):
                outputs = self.model(input_ids)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature
                
                # Get probabilities
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                
                # Sample from the distribution
                next_tokens = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_tokens], dim=1)
        
        # Decode all sequences
        generated_texts = []
        for seq in input_ids:
            text = self.tokenizer.decode(seq.tolist())
            generated_texts.append(text)
            
        return generated_texts

def main():
    # Initialize the model
    model = CustomModelInterface()
    
    # Example usage
    prompt = "My name is "
    outputs = model.generate(
        prompt=prompt,
        max_length=30,
        num_return_sequences=3,
        temperature=0.8
    )
    
    # Print results
    print(f"Prompt: {prompt}\n")
    print("Generated sequences:")
    for i, text in enumerate(outputs, 1):
        print(f"{i}. {text}\n")

if __name__ == "__main__":
    main()