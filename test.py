from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm


q1 = torch.tensor([2.,3],requires_grad = True)
q2 = torch.tensor([4.,7],requires_grad = True)
q3 = torch.tensor([2.,3],requires_grad = True)
q4 = torch.tensor([4.,7],requires_grad = True)
q5 = torch.tensor([2.,3],requires_grad = True)
q6 = torch.tensor([4.,7],requires_grad = True)
e1 = q1+q2
e2 = e1+q3
e3 = e2+q4
e4 = e3+q5
e7 = q1+e4
e8 = e4+q6
e9 = e7+e8
e10= e9+q6
e11 = e10+e9

e22 = torch.tensor([99,100.])
e11.backward(e22)
print(q1.grad)
