import torch 

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

print(f"we are using {device}")

random_seed = 4242
# to use seed, use it where generator is needed: torch.Generator().manual_seed(424242)
# we do not need to put random generator to GPU


if __name__ == "__main__":
    from torch.utils.data import random_split

    first = [ t.item() for t in random_split(torch.arange(0,20).to(device), [10,10], generator = torch.Generator().manual_seed(424242))[0] ]
    print(first)
    first = [ t.item() for t in random_split(torch.arange(0,20).to(device), [10,10], generator = torch.Generator().manual_seed(424242))[0] ]
    print(first)
    