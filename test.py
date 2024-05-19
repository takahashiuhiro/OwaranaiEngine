from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

q = torch.Tensor([1,2,3,4,5,6,7,8,9,10,11,12.])
q = q.view([2,3,2])
e = q.split([2,1],1)
for asdasd in e:
    print(asdasd)