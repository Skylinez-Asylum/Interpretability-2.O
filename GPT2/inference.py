from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(prompt, temperature=0.7, max_length=200):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token 
    model = AutoModelForCausalLM.from_pretrained("gpt2")
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
    
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    prompt = """Read the question and give an honest answer. Your answers should not include any unethical, racist, sexist, dangerous, or illegal content. If the question is wrong, or does not make sense, accept it instead of giving the wrong answer.
Question: Who is Bruce Wayne?
Answer:"""
    response = generate_response(prompt)
    print(response)