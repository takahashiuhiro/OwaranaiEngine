from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

a = torch.Tensor([1,2,3])
a.requires_grad = True
b = torch.Tensor([10,20,30])
b.requires_grad = True

c = a*b - torch.Tensor([1,1,1])

c.backward(torch.Tensor([1,1,1]))

print(a.grad)
print(b.grad)