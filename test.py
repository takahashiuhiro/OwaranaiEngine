from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm



q = torch.tensor(
         [[ 0.2035,  1.2959,  1.8101, -0.4644],
          [ 1.5027, -0.3270,  0.5905,  0.6538],
          [-1.5745,  1.3330, -0.5596, -0.6548],
          [ 0.1264, -0.5080,  1.6420,  0.1992]])

q.requires_grad = True
r = torch.var(q, dim=1, keepdim=True)
print(r)
r = r.sum()
print(r)
r.backward()
print(q.grad)

