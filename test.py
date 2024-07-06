from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

test_shape = [2,3,4,5]
summ=1
for a in test_shape:summ*=a
res = []
for a in range(1,summ+1):res.append(a)

q = torch.Tensor(res)
q = q.view(test_shape)

w = torch.Tensor(res).view(test_shape).transpose(1,2)
q.requires_grad = True

qq = q.transpose(1,2)

qq.backward(w)

print(qq)
print(q.grad)

