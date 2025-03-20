from customGPT2.model import load_model
from warnings import filterwarnings
import tiktoken
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from GPT2.inference import generate_response

# Suppress warnings and set precision for better performance
filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# Define configurations for custom models
@dataclass
class GPTConfig6:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768

@dataclass
class GPTConfig12:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# Load the four models
print('Loading models...')

# Model 1: gpt2

# Model 2: Custom GPT-2 with 12 heads, fine-tuned
path2 = "/home/arjun/Desktop/GitHub/Interpretability-2.O/customGPT2FT/save_states/12headFT9epoch-best.pt"
model2 = load_model(path2, GPTConfig12)
model2.to('cuda')

# Model 3: pretrained CustomGPT2
path3 = "customGPT2/save_states/12headNFT900k.pt"
model3 = load_model(path3, GPTConfig12)
model3.to('cuda')

# Model 4: Custom GPT-2 from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("Arjun-G-Ravi/chat-GPT2")
tokenizer.pad_token = tokenizer.eos_token
model4 = AutoModelForCausalLM.from_pretrained("Arjun-G-Ravi/chat-GPT2")
model4.to('cuda')

print('All models loaded\n')

# Inference function for custom models
def inference_custom(model, inp: str, max_length: int = 100, num_return_sequences: int = 1):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(inp)
    tokens = torch.tensor(tokens, dtype=torch.long, device='cuda')
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens
    while x.size(1) < max_length:
        with torch.inference_mode():
            logits, _ = model(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            xcol = torch.gather(topk_indices, 1, ix)
            x = torch.cat((x, xcol), dim=1)
    outs = []
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decode = enc.decode(tokens)
        outs.append(decode)
    return outs

# Inference function for the Transformers model
def inference_transformers(model, tokenizer, inp: str, max_length: int = 100, num_return_sequences: int = 1):
    encoding = tokenizer(
        inp,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True
    )
    input_ids = encoding.input_ids.to('cuda')
    attention_mask = encoding.attention_mask.to('cuda')
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
    )
    outs = []
    for i in range(num_return_sequences):
        decode = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        outs.append(decode)
    return outs

# Function to extract the response from the generated output
def extract_response(full_output: str) -> str:
    marker = "### Response:\n"
    response_start = full_output.find(marker) + len(marker)
    if response_start != -1:
        return full_output[response_start:].strip()
    return full_output.strip()

# Model showdown function
def model_showdown(question: str):
    # Unified prompt format for all models
    prompt = f'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n {question}\n\n### Input:\n\n### Response:
'''
    
    # Generate responses from each model
    response1 = generate_response(prompt, max_length=100)
    
    output2 = inference_custom(model2, prompt, max_length=100, num_return_sequences=1)[0]
    response2 = extract_response(output2)
    
    output3 = inference_custom(model3, prompt, max_length=100, num_return_sequences=1)[0]
    response3 = extract_response(output3)

    prompt = f'''
Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question: {question}
Answer: '''
    
    output4 = inference_transformers(model4, tokenizer, prompt, max_length=100, num_return_sequences=1)[0]
    response4 = extract_response(output4)
    
    # Display the results
    print(f"\nQuestion: {question}")
    print("----------------------------------------")
    print("Model 1: GPT2:")
    print(response1)
    print("----------------------------------------")
    print("Model 2  GPT2 FT:")
    print(response4)
    print("----------------------------------------")
    print("Model 3: Custom GPT2 Pretrained:")
    print(response3)
    print("----------------------------------------")
    print("Model 4: Custom GPT2 FT:")
    print(response2)
    print("----------------------------------------")

# Main execution
if __name__ == "__main__":
    # question = input("Enter your question: ")
    question = 'Generate a list of adjectives that describe a person as brave.'

    model_showdown(question)