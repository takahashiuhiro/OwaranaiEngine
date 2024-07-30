from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

print(torch.randint(5, (3,), dtype=torch.int64))