from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

ggq = torch.nn.MSELoss()

q = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12.])


g = q.view([2,3,2])
g.requires_grad = True


r = torch.exp(g)
r=r.sum()

r.backward( torch.tensor(1, dtype=torch.float))
print(r)
print(g.grad)

