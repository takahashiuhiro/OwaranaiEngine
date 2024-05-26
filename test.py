from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

def cc(a):
    b = torch.exp(-2*a)
    return (1-b)*((1+b)**(-1))

ggq = torch.nn.MSELoss()

q = torch.Tensor([1,2,3,4,5,6.])
w = torch.Tensor([1,2,3,4,5,6,7,8.])
e = torch.Tensor([1,2,3,4,5,6,7,8,9,10.])


q = q.view([2,3])
q.requires_grad = True

#w = w.view([2,4])
#w.requires_grad = True
#
#e = e.view([2,5])
#e.requires_grad = True
#
#r = torch.cat([q,w,e],1)
#print(r)
#r=r.sum()
#
#r.backward()
#print(r)
#print(q.grad)
#print(w.grad)
#print(e.grad)

#gg = torch.nn.GELU()
#r = gg(q).sum()
#r.backward()
#print(q.grad)

print(q.mean([0,1],True))
