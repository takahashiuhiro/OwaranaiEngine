from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm


    
q = torch.ones([2,3,4])

print(torch.tril(q,1000))