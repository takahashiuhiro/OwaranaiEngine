import torch

a =torch.zeros(2,2,3)
a.fill_(1)
a[0,1,1] = 1.5
a[1,0,2] = 8
a[1,1,0] = 18
print(a.min(dim = 2))