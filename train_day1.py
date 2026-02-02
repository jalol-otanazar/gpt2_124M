import os
import inspect
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch.cuda.amp import autocast, GradScaler

@dataclass
class GPTConfig:
    block_size : int = 1024
    vocab_size : int = 50257
    n_head : int = 12
    n_layer: int = 12
    n_embd : int = 768

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.std = 1.0
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size) )

        self.n_embd = config.n_embd
        self.n_head = config.n_head
    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, nh, T, hs
        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # B, nh, T, T
        # att = att.masked_fill(self.bias[:T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1) # B, nh, T, T
        # y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) = (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.std = 1.0
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self,  config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module): 

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.transformer.wte.weight = self.lm_head.weight # weight tying
        self.apply(self._weight_init)
    
    def _weight_init(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'std'):
                std *= (2 * self.config.n_layer) **-0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_embd = self.transformer.wpe(pos) # (   T, n_embd)
        tok_embd = self.transformer.wte(idx) # (B, T, n_emb)
        x = tok_embd + pos_embd # (B, T, n_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size), targets shape (B, T)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {'param': decay_params, 'weight_decay': weight_decay},
            {'param': nodecay_params, 'weight_decay': 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
import tiktoken

class DataLoader:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f'processing {len(tokens)} tokens')
        print(f'1 epoch = {len(tokens) // (B * T)}')

        self.current_pos = self.B * self.T *  process_rank

    def get_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current_pos : self.current_pos + B*T+1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)

        self.current_pos += B * T * self.num_processes # update current pos go to the next batch 
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_pos = self.B * self.T *  self.process_rank

        return x, y
    
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
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
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")

# Enable TF32 for matrix multiplications (slight speedup on T4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
B = 16
T = 1024
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)
total_batch_size = 524288
assert total_batch_size % (B * T * ddp_world_size) == 0 
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f'total desired batch size {total_batch_size}')
print(f'grad_accum_steps {grad_accum_steps}')

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    # warmup steps lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # the bottom lr:
    if step > max_steps:
        return min_lr
    # cosine decay, in between step

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



optimizer = model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

scaler = GradScaler(device=device) # fp 16 precision

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    loss_accum = 0.0
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
         # Forward pass in FP16
        with autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # logits, loss = model(x, y)  #fp32 forward pass
        # Backward pass with gradient scaling
        if ddp:
            model.require_backward_grad_sync = (mini_step == grad_accum_steps - 1)
        scaler.scale(loss).backward()
        # loss.backward() FP32 backward pass
        # Optimizer step with scaling
        # Gradient clipping
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    scaler.step(optimizer)
    scaler.update()
    # optimizer.step() # fp32 updating
    torch.cuda.synchronize()
    t1 = time.time()
    total_tokens_size = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    dt = (t1-t0) * 1000
    tokens_per_sec = total_tokens_size / (t1 - t0)
    if ddp:
        print(f'step {step} loss {loss_accum.item():.4f} {dt:.4f} ms {tokens_per_sec:.4f} tok/s')
    
if ddp:
    destroy_process_group()

