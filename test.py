from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

q1 = torch.ones([1,3,1,2])
q2 = torch.ones([3,1,2,1])*2


q1.requires_grad = True
q2.requires_grad = True


e1 = q1+q2

e1.backward(e1)
print(e1)
print(q1.grad)
print(q2.grad)