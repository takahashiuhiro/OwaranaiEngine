import torch

a =torch.zeros(1,1,3)
a[0,0,0] = 1
a[0,0,1] = 2
a[0,0,2] = 3
a.requires_grad = True

c = torch.nn.Softmax(2)(a)
#c = a*b
d = torch.Tensor(
    [[[
        1,2,1
    ]]]
)
c.backward(d)
#b = torch.nn.Softmax(dim = 0)(a)


#b.backward(torch.Tensor([1,0,3]))



#print(torch.exp(a))
#print(sofxres * (torch.Tensor([1,0,3]) - (sofxres * torch.Tensor([1,0,3])).sum() ))
print(d.shape)
print(a.grad)
