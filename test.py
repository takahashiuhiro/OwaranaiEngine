from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

res = []
for a in range(1,211):res.append(a)

q = torch.Tensor(res)
q = q.view([2,3,5,7])


print(q)
print(q.transpose(0,2).contiguous())

