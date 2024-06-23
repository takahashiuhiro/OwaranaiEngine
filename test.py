from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm



q = torch.Tensor([1,2,3,4,5,6,7,8,9.])
q=q.view([3,3])
q.requires_grad = True

e = torch.tril(q,1).sum()

e.backward()

print(q.grad)
print(e)