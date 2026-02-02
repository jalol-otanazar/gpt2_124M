import os
import time
import math
import torch
import inspect
import tiktoken
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    block_size : int = 1024
    n_head : int = 12
    n_layer : int = 12
    n_embd : int = 768

class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                             .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()        
        qkv = self.attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.std = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(self.config.n_layer)]),
            ln_f = nn.LayerNorm(self.config.n_embd),
        ))
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.transformer.wte.weight = self.lm_head.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'std'):
                nn.utils.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.utils.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.utils.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        tok_embd = self.transformer.wte(idx) #(B, T, C)
        pos_embd = self.transformer.wpe(pos) #(   T, C)
        x = tok_embd + pos_embd
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    
    def configure_optimizer(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.ndim >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.ndim < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay' : weight_decay},
            {'params': nodecay_params, 'weight_decay' : 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f'num decay_param tensors {len(decay_params)} of which total of {num_decay_params} parameters')
        print(f'num nodecay_param tensors {len(nodecay_params)} of which total of {num_nodecay_params} parameters')
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = device == 'cuda' and fused_available
        print(f'using fused AdamW: {use_fused}')
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

class DataLoader:
    
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        enc = tiktoken.get_encoding('gpt2')
        with open('input.txt', 'r') as f:
            text = f.read()
        tokens = enc.encode(text)
        self.tokens =  torch.tensor(tokens)
        print(f'loading {len(self.tokens)} tokens')
        print(f'1 epoch = {len(self.tokens) / (B*T)}')
        self.current_pos = B * T * process_rank
    
    def get_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current_pos: self.current_pos + B*T+1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)

        self.current_pos += B * T * self.num_processes
        if self.current_pos + (B*T*self.num_processes+1) > len(self.tokens):
            self.current_pos = B * T * self.process_rank
        return x, y

# torchrun --standalone --nproc_per_node=8 train_gpt2.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 
if ddp:
    init_process_group(backend='nccl')
    assert torch.cuda.is_available()
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1 
    master_process = True
    device = device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = DDP(model)
raw_model = model.module if ddp else model

B = 4
T = 1024
total_batch_size  = 524288
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
print(f"grad_accum_steps = {grad_accum_steps}")
train_loader = DataLoader(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(step):
    # warmup step lr
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # the final lr
    if step > max_steps:
        return min_lr
    
    # in between lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# optimization
optimizer = raw_model.configure_optimizer(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    loss_accum = 0.0
    optimizer.zero_grad()
    for mini_step in range(grad_accum_steps):
        x, y = train_loader.get_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = raw_model(x, y)
        loss /= grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            raw_model.require_backward_grad_sync(mini_step == grad_accum_steps - 1)    
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = nn.utils.clip_grad_norm_(raw_model.parameters(), 1.0)
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    total_num_tokens = train_loader.B * train_loader.T * ddp_world_size * grad_accum_steps
    dt = (t1 - t0) * 1000
    tok_per_sec = total_batch_size / (t1 - t0)
    print(f"step {step}, loss {loss_accum.item():.4f} ")

if ddp:
    destroy_process_group()
        


