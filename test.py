import torch

a =torch.zeros(2,4)
a.fill_(1)
a.requires_grad = True

b = torch.nn.Softmax(dim = 1)(a)

c = torch.zeros(2,4)
c[0,2] = 0.5

b.backward(c)

print(b)
print(a.grad)