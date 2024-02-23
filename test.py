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

#tt = torch.nn.LayerNorm([4],elementwise_affine  = True)
#h = torch.nn.GELU()(a)


#h.backward(c)

#print(a.grad)
#print(h)
#print(n.grad)
#print(h)
#print(ln(a))
#print(tt.bias)
#print(tt.weight)


#qwe = torch.nn.Linear(4,3, bias=True)

#d = a.view(8,2)
#
#
#d.backward(c.view(8,2))
#
#print(a.grad)
#print(d)

#print(qwe.bias)

padding_idx = 1
#embedding = torch.nn.Embedding(4, 2, padding_idx=padding_idx)
#print(torch.Tensor([[1,2.],[3,4],[5,6],[7,8]]).shape)
embedding = torch.nn.Embedding.from_pretrained(torch.Tensor([[1,2.],[3,4],[5,6],[7,8]]),False,padding_idx=padding_idx)
#print(torch.ones(3),torch.Tensor([[[1,2.],[3,4],[5,6],[7,8]]]))
#with torch.no_grad():
#    embedding.weight[0] = torch.Tensor([[[1,2.]]])
#    embedding.weight[1] = torch.Tensor([[[3,4.]]])
#    embedding.weight[2] = torch.Tensor([[[5,6.]]])
#    embedding.weight[3] = torch.Tensor([[[7,8.]]])
wai_input = torch.LongTensor([2,2,0,3])
s = embedding(wai_input)
print(s)
s.backward(torch.Tensor([[1,1],[1,1],[1,1],[1,1]]))
print(embedding.weight.grad)