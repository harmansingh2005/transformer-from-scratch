import torch
from src.modules import positional_encoding

B, T, D = 2, 5, 16
x = torch.zeros(B, T, D)
y = positional_encoding(x)
print(y.shape)  # should be torch.Size([2, 5, 16])
print(torch.allclose(y[0], y[1]))  # True: same positions across batch