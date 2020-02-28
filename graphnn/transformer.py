import torch
from torch.nn import Module, Linear, Dropout, LayerNorm
import torch.nn.functional as F

from .multihead_attention import MultiheadAttention
from .residual import Residual

class Transformer(Module):
    r"""Transformer is made up of self-attn and feedforward network.
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
        self.self_attn = MultiheadAttention(d_model, nhead, bias=bias)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.activation = getattr(nn, activation)() if isinstance(activation, str) else activation

        self.learned_residual = learned_residual
        if learned_residual:
            self.residual = Residual(d_model, dim_feedforward)
            # self.residual = Residual(d_model // nhead, dim_feedforward)

        self.d_model = d_model
        self.nhead = nhead


    def forward(self, src, mask=None, return_weights=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
                (Batch Size, *, Seq Len, Embedding Size)
            mask: what to attend to (optional).
                (Batch Size, *, Seq Len, Seq Len) -> Binary Tensor, 1 to attend, 0 to exclude
            return_weights: True -> return MHA weights, else only attended src

        Out:
            src: Attented src
                (Batch Size, *, Seq Len, Embedding Size)
            attn_weights: attentions weights given to each edge
                (Batch Size, *, Num Heads, Seq Len, Seq Len)
        """

        if mask is None:
            mask = torch.ones(src.size()[:-1] + src.size()[-2:-1])

        src2, attn_weights = self.self_attn(src, src, src, mask=mask)
        # output.shape == (batch_size, vertices, d_model)

        # feed forward
        if self.learned_residual:
            src, residual_weights = self.residual(src, self.dropout(src2))
        else:
            src = src + self.dropout(src2)

        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)

        if return_weights:
            if self.learned_residual:
                return src, {'mha':attn_weights, 'res':residual_weights}
            return src, {'mha':attn_weights}
        return src
