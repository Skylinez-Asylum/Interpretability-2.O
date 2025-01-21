from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def get_last_layer_activations(prompt, temperature=0.7, max_length=100):
    tokenizer = AutoTokenizer.from_pretrained("Arjun-G-Ravi/chat-GPT2")
    model = AutoModelForCausalLM.from_pretrained("Arjun-G-Ravi/chat-GPT2")
    
    # Move model to evaluation mode
    model.eval()
    
    encoding = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    
    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask
    
    # Get the last hidden states by running the model with output_hidden_states=True
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
    # Get the last layer's hidden states
    last_layer_activations = outputs.hidden_states[-1]
    
    # Generate response for comparison
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
    )
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return {
        'activations': last_layer_activations,
        'generated_text': generated_text,
        'activation_shape': last_layer_activations.shape,
        'tokens': tokenizer.convert_ids_to_tokens(input_ids[0])
    }

if __name__ == "__main__":
    prompt = """
Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question: Difference between RAM and ROM.
Answer:"""
    
    results = get_last_layer_activations(prompt)
    
    print(f"Generated Response: {results['generated_text']}\n")
    print(f"Last Layer Activation Shape: {results['activation_shape']}")
    print("\nActivation values for first token:")
    print(results['activations'][0][0][:10])  # Print first 10 values for first token
    print("\nTokens:")
    print(results['tokens'])