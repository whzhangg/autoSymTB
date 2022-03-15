from input_data import point, result, node_feature
from rpnet import Polynomial
import torch

f = Polynomial()
optim = torch.optim.Adam(f.parameters(), lr=1)
print(f)
    
for step in range(100):
    optim.zero_grad()
    pred = f(point, node_feature)
    loss = (pred - result).pow(2).sum()
    print(pred)
    loss.backward()
    optim.step()


print("The network is fitted with the unrotated points:")
print("output vector:")
print(f(point, node_feature).detach().numpy())
print("target vector:")
print(result.numpy())