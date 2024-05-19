from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

q = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12.])
q.requires_grad = True

g = q.view([2,3,2])
e = g.split([1,1,1],1)
print(e[0])
print(e[1])
print(e[2])

r = e[0]*e[1]*e[2]

r = r.sum([1,2])
print(r)

