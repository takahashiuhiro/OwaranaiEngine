from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

q1 = torch.ones([2,3])
q2 = torch.ones([3,4])*2
q3 = torch.ones([4,5])*3
q4 = torch.ones([5,6])*4



q1.requires_grad = True
q2.requires_grad = True
q3.requires_grad = True
q4.requires_grad = True


e1 = q1@q2@q3@q4

e1.backward(e1)
print(e1)
print(q1.grad)
print(q2.grad)
print(q3.grad)
print(q4.grad)
