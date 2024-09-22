from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

a = [1,2,3,4.]
b = [4,3,2,1.]
dta = torch.Tensor(a).view([2,2])
dtb = torch.Tensor(b).view([2,2])
dta.requires_grad = True
dtb.requires_grad = True
dtc = (dta@dtb).sum()
print(dtc)
dtc.backward()
print(dta.grad)
print(dtb.grad)