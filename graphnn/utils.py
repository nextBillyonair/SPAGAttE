import copy
from torch.nn import ModuleList

# Clones Module N times, tie determines if all N are shared
def get_clones(module, N, tie=False):
    if tie:
        return ModuleList([module for i in range(N)])
    return ModuleList([copy.deepcopy(module) for i in range(N)])


# # Give size
# def generate_square_subsequent_mask(sz):
#     mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
#     mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
#     return mask
