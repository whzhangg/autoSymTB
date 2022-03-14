import torch
from e3nn.o3 import FullyConnectedTensorProduct, Irreps

# this script basically test if an arbitrary vector can be fitted to 
# by a tensor product

target_ir = Irreps('1x2e')
target = target_ir.randn(-1, normalization="component")  # random target

input_ir = (Irreps.spherical_harmonics(3) * 2).sort().irreps.simplify()
inputs = input_ir.randn(-1, normalization='norm')  # random inputs

rotate = torch.tensor([0.1, 0.1, 0.1], dtype = float)  # some rotation
rot_out = target_ir.D_from_angles(*rotate)  # the above rotation of the output vector space
rot_in = input_ir.D_from_angles(*rotate)  # the above rotation of the input vector space

rotated_in = torch.einsum('ij,j -> i', rot_in.float(), inputs)

f = FullyConnectedTensorProduct(input_ir, input_ir, target_ir)

result = f(inputs, inputs)
rot_result = torch.einsum('ij,j -> i', rot_out.float(), result)

# this shows that rotation work as expected

optim = torch.optim.Adam(f.parameters(), lr=1)
for i in range(100):
    pred = f(inputs, inputs)
    if i % 20 == 0:
        print(pred.detach().numpy())
    loss = (pred - target).pow(2).sum()

    optim.zero_grad()
    loss.backward()
    optim.step()
print("target results")
print(target)

print("")
print("prediction with rotated inputs")
print(f(rotated_in, rotated_in))
print("rotated target output")
print(torch.einsum('ij,j -> i', rot_out.float(), pred))