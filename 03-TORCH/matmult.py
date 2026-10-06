# use `pip install asitop` and `asitop` to monitor gpu usage in apple silicon 

import torch

mps_device = torch.device("mps")

dim = 512
normalizer = dim ** 0.75

# on GPU
#
rand_mat_1 = torch.rand((dim, dim)).to(mps_device) / normalizer
rand_mat_2 = torch.rand((dim, dim)).to(mps_device) / normalizer

# on CPU
#
# rand_mat_1 = torch.rand((dim, dim)) / normalizer
# rand_mat_2 = torch.rand((dim, dim)) / normalizer

# some mult operations

mult_mat_1 = rand_mat_1 @ rand_mat_2

# do lots of matrix multiplications in order for computation
# to outweigh RAM->GPU copying and get a GPU speedup
#
powmat1 = (
    mult_mat_1 @ mult_mat_1 @ 
    mult_mat_1 @ mult_mat_1 @ 
    mult_mat_1 @ mult_mat_1 @ 
    mult_mat_1 @ mult_mat_1 @
    mult_mat_1 @ mult_mat_1 @
    mult_mat_1 @ mult_mat_1 
)

# note:rand_mat_1 * rand_mat_2 seems to not use GPU for computation

# should be equal by permutation symmetry
tr1 = torch.trace(powmat1)

print(tr1)