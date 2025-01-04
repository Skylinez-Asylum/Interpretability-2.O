from model import load_model
from warnings import filterwarnings
import tiktoken
import torch
from torch.nn import functional as F 
filterwarnings('ignore')

def inference(model,inp:str,max_length:int = 50,num_return_sequences:int =1):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(inp)
    tokens = torch.tensor(tokens,dtype=torch.long, device='cuda')
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
    x = tokens
    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits, _ = model(x)  # Unpack the tuple (logits, loss)
            logits = logits[:, [-1], :]  # Take the last time step logits (shape: [batch_size, 1, vocab_size])
            probs = F.softmax(logits, dim=-1)  # Apply softmax to get probabilities (shape: [batch_size, 1, vocab_size])
            
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # Get top-k probabilities and their indices
            topk_probs = topk_probs.squeeze(1)  # Remove the time dimension (shape: [batch_size, top_k])
            topk_indices = topk_indices.squeeze(1)  # Same for indices (shape: [batch_size, top_k])
            
            ix = torch.multinomial(topk_probs, 1)  # Sample from top-k probabilities (shape: [batch_size, 1])
            xcol = torch.gather(topk_indices, 1, ix)  # Get the corresponding indices (shape: [batch_size, 1])
            x = torch.cat((x, xcol), dim=1)  # Concatenate the new token to the sequence (shape: [batch_size, seq_len+1])

    outs=[]
    for i in range(num_return_sequences):
        tokens = x[i,:max_length].tolist()
        decode = enc.decode(tokens)
        outs.append(decode)

    return outs


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')  # use tf32 <- felt this gives worse answers sometims
    path = r"customGPT-2/save_states/state_step555000.pt"
    print('Loading model...')
    model = load_model(path)
    model.to('cuda')
    print('Model loaded')
    inp= "a train is"

    out = inference(model,inp,30,10)
    print(out)