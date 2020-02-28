import torch
import math
import torch.nn.functional as F
from torch.nn import Module, Dropout, Linear
from torch.nn.init import kaiming_uniform_

from .transformer import Transformer
from .utils import get_clones

# if needed for positional features
class PositionalEncoding(Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)


class NodeTransformer(Module):
    r"""Node Transformer is made up of self-attn and feedforward network.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, bias=True, learned_residual=False):
        super().__init__()
        self.transformer = Transformer(d_model, nhead, dim_feedforward, dropout, activation, bias, learned_residual)


    def forward(self, node, mask=None, return_weights=False):
        r"""Pass the input through the encoder layer.

        Args:
            node: the sequence to the encoder layer (required).
                (Batch Size, Vertices, Seq Len)
            mask: what to attend to (optional).
                (Batch Size, Vertices, Seq Len, Seq Len) -> Binary Tensor, 1 to attend, 0 to exclude
            return_weights: True -> return MHA weights, else only attended node

        Out:
            node: Attented node
                (Batch Size, Verticies, Seq Len, Embedding Size)
            attn_weights: attentions weights given to each edge
                (Batch Size, Vertices, Num Heads, Seq Len, Seq Len)
        """

        node, attn_weights = self.transformer(node, mask, True)

        if return_weights:
            return node, attn_weights
        return node


# N Transformers Stacked
class TransformerNetwork(Module):

    def __init__(self, transformer, n_layers=2, tie=False):
        super().__init__()
        self.layers = get_clones(transformer, n_layers, tie)
        self.n_layers = n_layers
        self.reset_parameters()

    def forward(self, embeddings, mask=None, return_weights=False):
        # in == (batch, verticies, seq, embed), adj_matrix == (batch, verticies, seq, seq)
        attn_weights = []

        for i in range(self.n_layers):
            embeddings, attn = self.layers[i](embeddings, mask, True)
            attn_weights.append(attn)

        # attn_weights shape == (B, V, H, S, S)
        # embedding shape == (B, V, S, E)
        if return_weights:
            return embeddings, attn_weights
        return embeddings

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                kaiming_uniform_(p)


# # Simple Feed Forward Layer
# class NodeEmbeddingNetwork(Module):
#
#     def __init__(self, input_dim, output_dim=2048, dim_feedforward=2048, num_layers=1, dropout=0.1, activation=F.relu, bias=True):
#         super().__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.activation = getattr(nn, activation)() if isinstance(activation, str) else activation
#         self.layers = ModuleList()
#         self.layers.append(Linear(input_dim, output_dim if num_layers == 1 else dim_feedforward))
#         for i in range(1, num_layers):
#             self.layers.append(Linear(dim_feedforward, dim_feedforward if i+1 != num_layers else output_dim))
#
#
#     def forward(self, node_features):
#         # in.shape == (batch, input_dim)
#         for layer in self.layers:
#             node_features = self.activation(layer(node_features))
#         return node_features # (batch, output_dim)




# EOF
