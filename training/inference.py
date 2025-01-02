import tiktoken
import torch
import torch.nn as nn
from torch.nn import functional as F 

def inference(model,inp:str,max_length:int = 50,num_return_sequences:int =1):
    model.eval()
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(inp)
    tokens = torch.tensor(tokens,dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences,1)
    x = tokens
    torch.manual_seed(42)
    while x.size(1) < max_length:
        with torch.no_grad():
            logits = model(x)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            topk_probs,topk_indices = torch.topk(probs,50,dim=-1)
            ix = torch.multinomial(topk_probs,1)
            xcol = torch.gather(topk_indices,-1,ix)
            x = torch.cat((x,xcol),dim=1)
    outs=[]
    for i in range(num_return_sequences):
        tokens = x[i,:max_length].tolist()
        decode = enc.decode(tokens)
        outs.append(decode)
    return outs