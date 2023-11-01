import torch
import torch.nn as nn

N, H,W = 2, 2,3
input1 = torch.rand(N, H,W)
# Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
# as shown in the image below
layer_norm = nn.LayerNorm([W])
output = layer_norm(input1)

def ln(inpt):
    im = inpt.mean((-1))#
    im = im.reshape((N,H,1))
    fz = inpt - im#
    var = (inpt - im)**2#
    var = var.mean((-1))#
    var = var.reshape((N,H,1))
    return fz/((torch.Tensor([1e-5]) + var)**0.5)#

print(ln(input1))
print(output)
print(((output - ln(input1))**2).sum())
