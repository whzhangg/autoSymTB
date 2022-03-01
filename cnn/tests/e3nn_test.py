import torch
from e3nn import o3, nn
from e3nn.math import soft_one_hot_linspace
from torch_scatter import scatter

def run_example_convolution():

    def find_edges(pos, rmax) -> list:
        result = []
        for i, x in enumerate(pos):
            for j, y in enumerate(pos):
                if i == j:
                    continue
                if torch.norm(x-y) < rmax:
                    result.append((i,j))
        return result

    # 
    print("This example script perform the task to evalulate the point convolution")
    print("")
    print("[ Step 1 ] Generating a set of point in the unit sphere and identify neighbors")

    # 100 point in the unit sphere, we find all the edges connecting their neigbors 
    npoint = 100
    rmax = 1.0
    pos = torch.randn((npoint, 3))
    from_to = find_edges(pos, rmax = rmax)
    r_ab_from = [i for (i,_) in from_to]
    r_ab_to   = [j for (_,j) in from_to]
    r_ab = pos[r_ab_to] - pos[r_ab_from]
    print("-> Generate {:d} points ...".format(npoint))
    print("-> average number of neigbhors: {:.1f} in the cutoff distance {:.1f}".format(len(r_ab)/npoint, rmax))
    # 
    # 
    print("")
    print("[ Step 2 ] Generating the Irreducible representations of the input, output")
    print("-> The input representation:  10x0e + 10x1e")
    print("-> The output representation: 20x0e + 10x1e")
    irreps_in  = o3.Irreps('10x0e + 10x1e')
    irreps_out = o3.Irreps('20x0e + 10x1e')
    f_in = irreps_in.randn(npoint, -1)       # -1 will be replaced by self.dim, therefore it return a tensor with shape [ 100 * 40 ]
    #
    #
    print("-> Spherical harmonic filter lmax = 2")
    irreps_filter = o3.Irreps.spherical_harmonics(lmax = 2)  # this returns irreducible representations of the spherical harmonics
    sh = o3.spherical_harmonics(irreps_filter, r_ab,                            # it calculates the spherical harmonics lmax = 2 gives 9 spherical harmonics
                                normalize=True, normalization='component')  # the output shape is [len(r_ab) * 9]       
    # 
    #
    print("-> Preparing Fully connected Tensor product ")
    tp = o3.FullyConnectedTensorProduct(irreps_in, irreps_filter, irreps_out, shared_weights = False)
    print(f">>> {tp}")      # the tensor product has 400 elements, each we can associate some weight parameters
    #
    #
    print("")
    print("[ Step 3 ] Creating Linear weights from MLP")
    print("-> Creating the linear weights from the distance |r_ab| and a MLP")
    nbasis = 10
    dist_embedding = soft_one_hot_linspace(
        r_ab.norm(dim=1), start = 0, end = rmax, number = nbasis, basis = 'smooth_finite', cutoff=True
    )   # create a feature vector on distance
    #
    print(f"-> The fully connected MLP has units {nbasis} * {20} * {tp.weight_numel}, ")
    print(f"   {tp.weight_numel} corresponds to the weights needed for linear combination of tensor product")
    fc = nn.FullyConnectedNet([nbasis, 20, tp.weight_numel])
    weight = fc(dist_embedding)                                   # shape [ len(r_ab) * 400 ]
    #
    #
    print("-> calculating the tensor product and the final result")
    print(f_in[r_ab_from].dtype)
    print(sh.dtype)
    print(weight.dtype)
    summand = tp(f_in[r_ab_from], sh, weight)                     # shape [ len(r_ab) * 50 ], 50 come from 20x0e + 10x1e
    #
    #
    f_out = scatter(summand, torch.tensor(r_ab_to), dim=0, dim_size=npoint)
    print("")
    print(f"Final result has the shape {f_out.shape}")

def run_example_tensorproduct():
    tp = o3.FullTensorProduct(irreps_in1='2x0e + 3x1o',
                              irreps_in2='5x0e + 7x1e')
    print("This is an example of full tensor product")
    print(f"  {tp}")

    tp = o3.FullyConnectedTensorProduct(irreps_in1='2x0e + 3x1o',
                                        irreps_in2='5x0e + 7x1e',
                                        irreps_out='15x0e + 3x1e')

    print("\nThis is an example of fully connected tensor product")
    print(f"  {tp}")
    print("   There are 192 weights, they correspond to fully connected part:")
    print("                          10x1e -> 15x0e (150 weights)")
    print("                          14x1e -> 3x1e  ( 42 weights)")
    print("\nThis example shows how Fully connected tensor product with weights work")
 

run_example_convolution()