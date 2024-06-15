# train script
# simple launch command:
# python train_gpt2.py
# DDP launch command:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py
# ----------------------------------------------------------------------------
import math
import os
import time

import tiktoken
import torch
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from gpt2_modules import GPT, GPTConfig
from dataloaderlite import DataLoaderLite

# training settings
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

ampere_gpu = True
if ampere_gpu:
    torch.set_float32_matmul_precision("high") # use tf32 where possible

use_compile = False

# Original GPT training hyperparameters
# max_lr = 6e-4
# min_lr = max_lr * 0.1
# warmup_steps = 715
# max_steps = 19073

# Change slightly to to a toy training run
max_lr = 6e-4 * 3
min_lr = max_lr * 0.1
warmup_steps = 100
max_steps = 2000

total_batch_size = 2 ** 19 # ~0.5M
B = 16 # micro batch size
T = 1024 # sequence length

# set up DDP
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"    
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing, etc.
else:
    # non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps" # apple silicon
    print(f"Using device: {device}")

# Get data loader with gradient accumulation
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, split="train", process_rank=ddp_rank, num_processes=ddp_world_size)
val_loader = DataLoaderLite(B=B, T=T, split="val", process_rank=ddp_rank, num_processes=ddp_world_size)

# Create model
# model = GPT.from_pretrained("gpt2") # or load pretrained from HuggingFace
enc = tiktoken.get_encoding("gpt2")
model = GPT(GPTConfig(vocab_size=50304)) # increase from 50257 (original vocab size) to a "nice" number
model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contian the "raw" unwrapped model

if use_compile:
    model = torch.compile(model)

# Create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

# Optimize
def get_lr(it):
    # 1. Linear warmup
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2. If we surpassed max_steps, return min learning rate
    if it > max_steps:
        return min_lr
    # 3. In between; use cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device, master_process=master_process)

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 2 ** 25 // (B * T * ddp_world_size)
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"Validation loss {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "config": raw_model.config,
                    "step": step,
                    "val_loss": val_loss_accum.item(),
                }
                torch.save(checkpoint, checkpoint_path)


    # TODO: once in a while evaluate hellaswag

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                if ampere_gpu:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits, loss = model(xgen)
                else:
                    logits, loss = model(xgen)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probs
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix)
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample{i}: {decoded}")
            
    # one step of optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ampere_gpu:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits, loss = model(x, y)
        else:
            logits, loss = model(x, y)
        loss /= grad_accum_steps # normalize
        loss_accum += loss.detach()

        if ddp:
            # only synchronize if we are in the final iteration of the micro steps
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    if torch.cuda.is_available():
        # wait for all cuda processes to finish to get accurate timing
        torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tok_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt * 1000:.2f} | tok/sec: {tok_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
