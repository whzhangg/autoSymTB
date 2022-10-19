import torch, typing
from torch_geometric.data import Data as PygData

__all__ = ["train_model"]

def train_model(func, pygdatas: typing.List[PygData], lr = 1e-2, nstep: int = 300):
    optim = torch.optim.Adam(func.parameters(), lr=lr)
    torch_dtype = pygdatas[0].x.dtype
    torch_device = pygdatas[0].x.device
    for step in range(nstep):
        for pygdata in pygdatas:
            optim.zero_grad()
            pred = func(pygdata)
            loss = (pred - pygdata.y).pow(2).sum()
            loss.backward()
            optim.step()

        if step % 10 == 0:
            with torch.no_grad():
                train_score = torch.zeros((1,)).to(device=torch_device, dtype=torch_dtype)
                for pygdata in pygdatas:
                    train_score += (func(pygdata) - pygdata.y).pow(2).sum()
                train_score /= len(pygdatas)
                print(f"epoch {step:5d} | l2loss {train_score.item():<10.5f}")
            
            if torch.sqrt(train_score) < 1e-4:
                break
    return func