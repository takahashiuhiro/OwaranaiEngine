import torch

a = torch.zeros(1,3)
a.fill_(1.5)
a.requires_grad = True

b = torch.zeros(3,1)
b.fill_(1.2)
b.requires_grad = True



c = a@b

d = c@a

e = d@b

e.backward()


print(a.grad)
print(b.grad)