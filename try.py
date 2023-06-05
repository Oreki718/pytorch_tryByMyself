import torch

a = torch.randn(4, 4)
print(a)
b = torch.max(a, 1)
print(b[1])