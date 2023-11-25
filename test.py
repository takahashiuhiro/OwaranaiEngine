import torch
import torch.nn as nn
import math


def ln(inpt):
    im = inpt.mean((-1))#
    im = im.reshape((4,1))
    fz = inpt - im#
    var = (inpt - im)**2#
    var = var.mean((-1))#
    var = var.reshape((4,1))
    return fz/((torch.Tensor([1e-5]) + var)**0.5)#

#print(ln(input1))
#print(output)
#print(((output - ln(input1))**2).sum())


#a = torch.tensor([[1.,2,3]],requires_grad = True)
##b= a.pow(7)
#c = a.var(1,unbiased = False,keepdims = True)
#c.backward(torch.tensor([[10.]]))
#
#print(c)
#print(a.grad)

a = torch.tensor(
    [[ 0.2035,  1.2959,  1.8101, -0.4644],
     [ 1.5027, -0.3270,  0.5905,  0.6538],
     [-1.5745,  1.3330, -0.5596, -0.6548],
     [ 0.1264, -0.5080,  1.6420,  0.1992]],requires_grad = True)



c= torch.tensor(
    [[ 0.2035,  1.2959,  1.8101, -0.4644],
     [ 1.5027, -0.3270,  0.5905,  0.6538],
     [-1.5745,  1.3330, -0.5596, -0.6548],
     [ 1,1,1,1.]])
#z = torch.var(a, dim=[0,1], keepdim=True)

def varr(q):
    r = (q - q.mean(dim=[0,1], keepdim=True))**2
    tt = r.shape[1]*r.shape[0]
    r = r.sum(dim=[0,1], keepdim=True)
    r /= tt
    return r

#h = torch.var(a, dim=[0,1], keepdim=True, unbiased = False)
#h = varr(a)
#h.backward(torch.tensor([[1.]]))
#c = torch.ones((4,4))

tt = torch.nn.LayerNorm([4],elementwise_affine  = True)
h = torch.nn.Tanh()(a)


h.backward(c)

print(a.grad)
print(h)
#print(n.grad)
#print(h)
#print(ln(a))
#print(tt.bias)
#print(tt.weight)
