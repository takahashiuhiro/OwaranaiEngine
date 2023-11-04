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

#print(ln(input1))
#print(output)
#print(((output - ln(input1))**2).sum())


a = torch.tensor([[[1.,2,3],[4,5,6]],[[7,8,9],[10,11,12]]],requires_grad = True)
b= a.mean([-2,-1],keepdim = True)
b.backward(torch.tensor([[[777]],[[666]]]))
print(a.grad)