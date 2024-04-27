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
e2 = torch.ones([3,3,2,2])
e2[1,2,1,0] = 55
e2[0,2,1,0] = 345
e2[0,1,0,1] = 95
e2[2,0,0,0] = 1082

q1.requires_grad = True
q2.requires_grad = True


e1 = q2@q1
#e1 = e1.sum([1,3])
#e1 = e1.view([1,-1])
e1 = nn.Softmax(dim=1)(e1)

e1.backward(e2)
print(e1)
print(e1.shape)
print(q1.grad)
print(q2.grad)