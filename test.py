import torch

a =torch.zeros(3)
a.fill_(2)
a[1] = 1
a.requires_grad = True


b = torch.nn.Softmax(dim = 0)(a)


b.backward(torch.Tensor([1,0,3]))



print(torch.exp(a))
print(torch.exp(a)/torch.exp(a).sum()*(torch.Tensor([1,0,3]) - (torch.exp(a)/torch.exp(a).sum()*torch.Tensor([1,0,3])).sum()))
print(a.grad)
