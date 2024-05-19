from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

q = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12.])


g = q.view([2,3,2])
g.requires_grad = True
e = g.split([1,1,1],1)

r = e[0]*e[1]*e[2]

r = r.sum(0)
r = r.sum(0)
r = r.sum(0)
r.backward()
print(r)
print(g.grad)

