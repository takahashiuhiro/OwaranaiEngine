from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

print(torch.arange(1, 2.5+1e-8, 0.5))