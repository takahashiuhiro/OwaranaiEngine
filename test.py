from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm

a = torch.Tensor([1,2,3])
b = torch.Tensor([[1],[2],[3]])

print(a+b)