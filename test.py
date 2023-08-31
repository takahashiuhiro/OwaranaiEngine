import torch

a =torch.zeros(2,3)
a.fill_(4)
a[1,0] = 1
a.requires_grad = True

b = torch.nn.Softmax(dim = 0)(a)


print(b)
