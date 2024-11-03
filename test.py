from torch import nn, Tensor
import torch.nn.functional as F
import torch
import math
from torch.autograd.variable import Variable
import typing
import random
import tqdm
import torch.optim as optim

import torch

def gaussian_cdf(x, mean=0.0, std=1.0, terms=3):
    """
    Compute the CDF of a normal distribution using a Taylor series approximation of the error function.
    
    Parameters:
    - x (torch.Tensor): The input value or tensor of values.
    - mean (float): The mean of the Gaussian distribution.
    - std (float): The standard deviation of the Gaussian distribution.
    - terms (int): The number of terms in the Taylor series for the error function.
    
    Returns:
    - torch.Tensor: The CDF value(s) for the input.
    """
    
    # Standardize the input to the standard normal form
    #z = (x - mean) / (std * torch.sqrt(torch.tensor(2.0)))
    z = x
    
    # Initialize erf_approx as a tensor with the same shape as z, filled with zeros
    erf_approx = torch.zeros_like(z)
    
    # Compute the Taylor series expansion of the error function
    for k in range(terms):
        coef = ((-1)**k * z**(2 * k + 1)) / (torch.factorial(torch.tensor(k)) * (2 * k + 1))
        erf_approx += coef
    
    # Compute the CDF using the approximation of the error function
    cdf = 0.5 * (1 + (2 / torch.sqrt(torch.tensor(torch.pi))) * erf_approx)
    
    return cdf

# Example usage
mean = 0.0
std = 1.0
terms = 3
x = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])  # Example input tensor
result = gaussian_cdf(x, mean=mean, std=std, terms=terms)
print(result)
