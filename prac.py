import torch
from torch import nn

LogSoftmax = nn.LogSoftmax(dim=1)

a = torch.tensor([[2.5, -2.1],[1.2, -0.8], [4, -4]], dtype= torch.float)
label = torch.tensor([[1, 0],[0, 1], [0, 1]], dtype=torch.float)

bs, cond_dim = a.size(0), a.size(1)

confidence = LogSoftmax(a)

print(confidence)
print(torch.exp(confidence))


log_ll = torch.bmm(confidence.view(bs, 1, cond_dim), label.view(bs, cond_dim, 1)).view(bs)

print(log_ll)


from aaa import dd