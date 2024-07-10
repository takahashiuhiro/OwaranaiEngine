from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm



q = torch.Tensor([1,2,3,4,5,6,7,8,9.])
q = q.view([3,3])

w = torch.Tensor([1,2,3,4,5,6,7,8,9.])
w = w.view([3,3])

q.requires_grad = True

mmsk = torch.ones([3,3],dtype = torch.bool).tril()

qq = q.masked_fill(mmsk, 99)

qq.backward(w)

print(qq)
print(q.grad)

