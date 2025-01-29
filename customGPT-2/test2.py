from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
import torch
from model import GPT, GPTConfig

# Define the custom configuration
class CustomGPTConfig(PretrainedConfig):
    model_type = "custom_gpt"
    
    def __init__(
        self,
        block_size=1024,
        vocab_size=50304,
        n_layer=12,
        n_head=6,
        n_embd=768,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

# Define the custom model
class CustomGPTModel(PreTrainedModel):
    config_class = CustomGPTConfig
    base_model_prefix = "transformer"
    
    def __init__(self, config):
        super().__init__(config)
        gpt_config = GPTConfig(
            block_size=config.block_size,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_embd=config.n_embd
        )
        self.transformer = GPT(gpt_config)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.post_init()
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Pass input_ids and labels to the transformer
        logits, loss = self.transformer(input_ids, labels)
        logits = self.lm_head(logits)

        # Return the causal LM outputs
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # Prepare inputs for the generate() method
        return {"input_ids": input_ids}

    def _tie_weights(self):
        # Tie weights between lm_head and transformer embeddings
        self.lm_head.weight = self.transformer.wte.weight


class CustomModelInterface:
    def __init__(self, model_name="Arjun-G-Ravi/Custom-GPT-555k"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Register the custom model
        AutoConfig.register("custom_gpt", CustomGPTConfig)
        AutoModelForCausalLM.register(CustomGPTConfig, CustomGPTModel)
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(model_name, ignore_mismatched_sizes=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def generate(self, prompt, max_length=50, num_return_sequences=1, temperature=1.0):
        # Encode the input text
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        
        # Generate output tokens
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_k=50,
                do_sample=True
            )
        
        # Decode all sequences
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]

def main():
    # Initialize the model
    print("Initializing model...")
    model = CustomModelInterface()
    
    # Example usage
    prompt = "My name is "
    print(f"\nGenerating text for prompt: '{prompt}'")
    
    outputs = model.generate(
        prompt=prompt,
        max_length=30,
        num_return_sequences=3,
        temperature=0.8
    )
    
    # Print results
    print("\nGenerated sequences:")
    for i, text in enumerate(outputs, 1):
        print(f"{i}. {text}\n")

if __name__ == "__main__":
    main()
