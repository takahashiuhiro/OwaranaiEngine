import torch

a =torch.zeros(1,3,3)
a.requires_grad = True

c = a.sum(1)
#c = a*b
d = torch.Tensor(
    [[
        1,2,26
    ]]
)
c.backward(d)
#b = torch.nn.Softmax(dim = 0)(a)


#b.backward(torch.Tensor([1,0,3]))



#print(torch.exp(a))
#print(torch.exp(a)/torch.exp(a).sum()*(torch.Tensor([1,0,3]) - (torch.exp(a)/torch.exp(a).sum()*torch.Tensor([1,0,3])).sum()))
print(d.shape)
print(a.grad)
