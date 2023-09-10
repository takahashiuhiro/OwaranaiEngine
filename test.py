import torch

a =torch.zeros(1,3,1)
a.fill_(2)
a[0,1,0] = 1
a.requires_grad = True

c = torch.broadcast_to(a, (3,3, 3))
#c = a*b
d = torch.Tensor(
    [
    [
        [1,2,3],
        [10,20,30],
        [100,200,300]
    ],
    [
        [1,2,3],
        [10,2000,30],
        [100,200,300]
    ],
    [
        [1,2,3],
        [10,2099999,30],
        [100,200,300666666666]
    ],
    ]
)
c.backward(d)
#b = torch.nn.Softmax(dim = 0)(a)


#b.backward(torch.Tensor([1,0,3]))



#print(torch.exp(a))
#print(torch.exp(a)/torch.exp(a).sum()*(torch.Tensor([1,0,3]) - (torch.exp(a)/torch.exp(a).sum()*torch.Tensor([1,0,3])).sum()))
print(a.grad)
print(d.sum(2).sum(0))
