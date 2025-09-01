import torch
from src.modules import PositionalEncoding

B, T, D = 2, 5, 16
x = torch.zeros(B, T, D)
pe = PositionalEncoding(D, dropout=0.0)
y = pe(x)
print(y.shape)  # should be torch.Size([2, 5, 16])
print(torch.allclose(y[0], y[1]))  # True: same positions across batch