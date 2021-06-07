import torch

a = torch.randn(4, 4)
print(a)

sa = torch.argsort(a, descending=True)
print(sa)
print(sa[:,:2])