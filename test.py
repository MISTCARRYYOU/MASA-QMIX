import numpy as np
import torch
b = torch.tensor([[6],[3],[1]])
c= torch.tensor([[1],[1],[0]])
b[c==1]=-6
print(b[c == 1], b)
