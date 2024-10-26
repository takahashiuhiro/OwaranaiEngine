from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

# With Learnable Parameters
m = nn.BatchNorm2d(2, affine=False)
input = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24.])
output = m(input.view([2,2,2,3]))

print(output)