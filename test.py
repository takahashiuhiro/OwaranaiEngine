import torch

a =torch.zeros(1,3)
a.fill_(2)
a[0,1] = 1
a.requires_grad = True

b =torch.zeros(1,3)
b.fill_(1)
#b[1] = 1
b.requires_grad = True

c = a*b

c.backward(torch.ones(1,3))
#b = torch.nn.Softmax(dim = 0)(a)


#b.backward(torch.Tensor([1,0,3]))



#print(torch.exp(a))
#print(torch.exp(a)/torch.exp(a).sum()*(torch.Tensor([1,0,3]) - (torch.exp(a)/torch.exp(a).sum()*torch.Tensor([1,0,3])).sum()))
print(a.grad)
