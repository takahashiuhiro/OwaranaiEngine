from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

dd = [1,2,3,4,5,math.e,7,8,math.pi,math.pi*2, math.e*3,math.e*4]
ds = [2,2,3]

qq = torch.Tensor(dd).view(ds)

tg = torch.Tensor(dd).view(ds)

qq.requires_grad = True

g = F.cross_entropy(qq,target = tg,reduction = "sum")

g.backward()

print(qq)
print(g)
print(qq.grad)