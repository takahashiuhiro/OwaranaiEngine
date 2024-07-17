from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

torch.set_printoptions(precision=8)

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

with torch.no_grad():  # 禁用梯度计算
    #gg = torch.ones(ly.attn.c_attn.weight.shape)
    #gg[1,2] = 2
    ly.attn.c_attn.weight = nn.Parameter(torch.Tensor([-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990,0.100614905357,0.553111672401,-0.346347510815,-0.217413812876,0.553031206131,-0.544376254082,0.515409946442,-0.068495750427,0.028762757778,-0.404539138079,-0.212561607361,0.120650708675,0.121993720531,-0.399396836758,-0.388366460800,0.246734738350,0.340686082840,-0.248414635658]).view(ly.attn.c_attn.weight.shape[-1], ly.attn.c_attn.weight.shape[-2]).transpose(-1,-2))
    ly.attn.c_attn.bias = nn.Parameter(torch.Tensor([-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990]).view(ly.attn.c_attn.bias.shape))
    ly.attn.c_proj.weight = nn.Parameter(torch.Tensor([-0.209389716387,0.313938617706,0.539439082146,-0.355767995119,-0.353367090225,-0.415830731392,0.534706830978,-0.217241346836,-0.011811017990]).view(ly.attn.c_proj.weight.shape[-1],ly.attn.c_proj.weight.shape[-2]).transpose(-1,-2))
    ly.attn.c_proj.bias = nn.Parameter(torch.Tensor([-0.209389716387,0.313938617706,0.539439082146]).view(ly.attn.c_proj.bias.shape))

    ly.mlp.c_fc.weight = nn.Parameter(torch.Tensor([-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750,-0.227874696255,0.251188397408,0.138770222664,-0.184441655874,0.460518240929,-0.026673555374,-0.278903633356,0.550726890564,-0.012399971485,-0.559777975082,0.311341106892,-0.393450826406,0.241994619370,0.348400592804,0.082190930843,0.361373245716,-0.124685674906,0.189551293850,-0.030301928520,-0.061172962189,-0.451027125120,0.195954144001,0.196026325226,0.253808140755]).view(ly.mlp.c_fc.weight.shape[-1],ly.mlp.c_fc.weight.shape[-2]).transpose(-1,-2))
    ly.mlp.c_fc.bias = nn.Parameter(torch.Tensor([-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750]).view(ly.mlp.c_fc.bias.shape))

    ly.mlp.c_proj.weight = nn.Parameter(torch.Tensor([-0.485680103302,-0.247496545315,-0.442987531424,0.217600405216,0.273247063160,0.219506263733,-0.026412069798,-0.502533078194,0.560715079308,0.428272485733,-0.426166892052,0.019910454750,-0.227874696255,0.251188397408,0.138770222664,-0.184441655874,0.460518240929,-0.026673555374,-0.278903633356,0.550726890564,-0.012399971485,-0.559777975082,0.311341106892,-0.393450826406,0.241994619370,0.348400592804,0.082190930843,0.361373245716,-0.124685674906,0.189551293850,-0.030301928520,-0.061172962189,-0.451027125120,0.195954144001,0.196026325226,0.253808140755]).view(ly.mlp.c_proj.weight.shape[-1],ly.mlp.c_proj.weight.shape[-2]).transpose(-1,-2))
    ly.mlp.c_proj.bias = nn.Parameter(torch.Tensor([-0.485680103302,-0.247496545315,-0.442987531424]).view(ly.mlp.c_proj.bias.shape))

ly = ly.cuda()

optimizer = optim.SGD(ly.parameters(), lr=0.01) 

inputx = [
    torch.Tensor([1,2,1,6,5,0.]).view([1,2,3]).cuda(),
    torch.Tensor([1,2,1,6,4,0.]).view([1,2,3]).cuda(),
]

outputx = [
    torch.Tensor([1,2,1,6,5,1.]).view([1,2,3]).cuda(),
    torch.Tensor([1,2,1,6,5,7.]).view([1,2,3]).cuda(),
]



#print(nn.MSELoss()(inputx[0], outputx[0]))

num_epochs = len(inputx)
criterion = nn.MSELoss()

for epoch in range(num_epochs+1):
    # 前向传播
    outputs = ly(inputx[epoch%2])
    loss = criterion(outputs, outputx[epoch%2])
    
    # 反向传播和优化
    optimizer.zero_grad()   # 清零梯度
    loss.backward()         # 反向传播
    optimizer.step()  

    print(outputs)
    #print(loss)

