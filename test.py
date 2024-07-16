from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v  = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # manual implementation of attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, -1e9)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class config(object):
    def __init__(self):
        self.dropout = 0
        self.n_embd =3
        self.n_head = 3
        self.bias = 1
        self.block_size = 40

cfg = config()
ly = Block(cfg)

#with torch.no_grad():  # 禁用梯度计算
#    gg = torch.ones(ly.c_attn.weight.shape)
#    gg[1,2] = 987
#    ly.c_attn.weight = nn.Parameter(gg)
#    ly.c_attn.bias = nn.Parameter(torch.ones(ly.c_attn.bias.shape)*0.3)
#    ly.c_proj.weight = nn.Parameter(torch.ones(ly.c_proj.weight.shape)*1.5)
#    ly.c_proj.bias = nn.Parameter(torch.ones(ly.c_proj.bias.shape)*2)


optimizer = optim.SGD(ly.parameters(), lr=0.01) 

inputx = [
    torch.Tensor([[[1,2,1.],[6,5,6.]]]),
    torch.Tensor([[[3,2,3.],[4,5,4.]]]),
]

outputx = [
    torch.Tensor([[[1,0,0.],[0,0,0.]]]),
    torch.Tensor([[[0,0,0.],[0,0,1.]]]),
]



#print(nn.MSELoss()(inputx[0], outputx[0]))

num_epochs = len(inputx)
criterion = nn.MSELoss()

for epoch in range(num_epochs):
    # 前向传播
    outputs = ly(inputx[epoch%2])
    loss = criterion(outputs, outputx[epoch%2])
    
    # 反向传播和优化
    optimizer.zero_grad()   # 清零梯度
    loss.backward()         # 反向传播
    optimizer.step()  

    print(outputs)
    print(loss)

