import numpy as np
from torch import nn

def num_parameters(self):
    return sum(np.prod(p.shape) for p in self.parameters())

nn.Module.num_parameters = property(num_parameters)

from .graph_attention_layer import GraphAttentionNetwork, GraphAttentionLayer
from .utils import get_clones
from .node_transformer import PositionalEncoding, NodeTransformer, TransformerNetwork
from .transformer import Transformer
from .multihead_attention import MultiheadAttention
from .residual import Residual
from .pairwise import PairwiseBilinear, PairwiseDot, PairwiseDistance
