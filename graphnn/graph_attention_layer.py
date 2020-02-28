import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.init import kaiming_uniform_

from .transformer import Transformer
from .utils import get_clones
# from .residual import Residual


class GraphAttentionLayer(Module):
    r"""GraphAttentionLayer is made up of self-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=F.relu, bias=True, learned_residual=False):
        super().__init__()
        self.gat = Transformer(d_model, nhead, dim_feedforward, dropout, activation, bias, learned_residual)


    def forward(self, nodes, adjacency_matrix=None, return_weights=False):
        r"""Pass the input through the encoder layer.

        Args:
            nodes: the sequnce to the encoder layer (required).
                (Batch Size, Vertices, Embedding Size)
            adjacency_matrix: the matrix of connections (optional).
                (Batch Size, Vertices, Vertices) -> Binary Tensor
                Note: Have such that Row -> Col.
            return_weights: True -> return MHA weights, else only attended nodes

        Out:
            nodes: Attented nodes
                (Batch Size, Vertices, Embedding Size)
            attn_weights: attentions weights given to each edge
                (Batch Size, Num Heads, Vertices, Vertices)
        """

        if adjacency_matrix is None:
            adjacency_matrix = torch.zeros(nodes.size(0), nodes.size(1), nodes.size(1), dtype=torch.float32)

        nodes, attn_weights = self.gat(nodes, adjacency_matrix, True)

        if return_weights:
            return nodes, attn_weights
        return nodes



# N Hop Version
class GraphAttentionNetwork(Module):

    def __init__(self, gal, n_hops=2, tie=False):
        super().__init__()
        self.graph = get_clones(gal, n_hops, tie)
        self.n_hops = n_hops
        self.reset_parameters()

    def forward(self, embeddings, adjacency_matrix=None, return_weights=False):
        # in == (batch, verticies, embed), adj_matrix == (batch, verticies, verticies)
        attn_weights = []

        for i in range(self.n_hops):
            embeddings, attn = self.graph[i](embeddings, adjacency_matrix, True)
            attn_weights.append(attn)

        if return_weights:
            return embeddings, attn_weights
        return embeddings

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                kaiming_uniform_(p)
