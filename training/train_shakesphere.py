import tiktoken
import torch
import torch.nn as nn
from config import GPTConfig
from model import GPT
from inference import inference
from dataloader import DataLoaderLite
import time
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import math

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50000
last_step = max_steps

def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps

    elif it > max_steps:
        return min_lr

    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


if __name__ == "__main__": 

   ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
   if ddp:
      # use of DDP atm demands CUDA, we set the device appropriately according to rank
      assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
      init_process_group(backend='nccl')
      ddp_rank = int(os.environ['RANK'])
      ddp_local_rank = int(os.environ['LOCAL_RANK'])
      ddp_world_size = int(os.environ['WORLD_SIZE'])
      device = f'cuda:{ddp_local_rank}'
      torch.cuda.set_device(device)
      master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
   else:
      # vanilla, non-DDP run
      ddp_rank = 0
      ddp_local_rank = 0
      ddp_world_size = 1
      master_process = True
      # attempt to autodetect device
      device = "cpu"
      if torch.cuda.is_available():
         device = "cuda"
      elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
         device = "mps"
      print(f"using device: {device}")

   device_type = "cuda" if device.startswith("cuda") else "cpu"

   total_batch_size = 524288
   B = 64 
   T = 1024
   assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
   grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
   if master_process:
       print(f"total desired batch size: {total_batch_size}")
       print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")
   
   train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size,split='train')
   val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")
   
   torch.set_float32_matmul_precision('high')

   #create model
   model = GPT(GPTConfig(vocab_size=50304))
   model.to(device)
   model = torch.compile(model)
   
   if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

   raw_model = model.module if ddp else model # always contains the "raw" unwrapped model
   
   optimizer = raw_model.configure_optimizers(weight_decay=0.1,learning_rate=6e-4,device_type=device_type,master_process=master_process)

   log_dir = "log"
   os.makedirs(log_dir, exist_ok=True)
   log_file = os.path.join(log_dir, f"log.txt")
   with open(log_file, "w") as f: # open for writing to clear the file
      pass
   
   for step in range(max_steps):

      t0 = time.time()
      if step % 100 == 0:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # training loop
      model.train()
      optimizer.zero_grad()
      
      loss_accum = 0.0
      for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
      
      if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

      norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      lr = get_lr(step)

      for param_group in optimizer.param_groups:
        param_group['lr'] = lr  

      optimizer.step()
      torch.cuda.synchronize()

      t1 = time.time()
      dt = t1 - t0
      tokens_processed = train_loader.B * train_loader.T* grad_accum_steps * ddp_world_size
      tokens_per_sec = tokens_processed/dt
      if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

if ddp:
    destroy_process_group()