import math
import torch
from torch.nn import Linear, Module

# Make generic for text or graph MHA
class MultiheadAttention(Module):

    def __init__(self, d_model, num_heads, hidden_size=None, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.hidden_size = d_model if hidden_size is None else hidden_size

        assert self.hidden_size % self.num_heads == 0, f"Hidden Dimension of model (hidden_size:{hidden_size}) must be divisible by num_heads ({num_heads})"

        self.depth = self.hidden_size // self.num_heads

        self.wq = Linear(d_model, self.hidden_size, bias=bias)
        self.wk = Linear(d_model, self.hidden_size, bias=bias)
        self.wv = Linear(d_model, self.hidden_size, bias=bias)

        self.dense = Linear(self.hidden_size, d_model, bias=bias)

    def scaled_dot_product_attention(self, query, key, value, mask):
        matmul_qk = torch.matmul(query, key.transpose(-2, -1))
        scaled_attention_logits = matmul_qk / math.sqrt(self.depth)

        mask = mask.unsqueeze(-3)
        scaled_attention_logits = scaled_attention_logits + (1 - mask) * (-1e10)
        another_mask = 1 - (~mask.bool()).all(-1, keepdim=True).float()

        attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
        attention_weights = attention_weights * another_mask

        output = torch.matmul(attention_weights, value) * another_mask

        return output, attention_weights

    def split_head(self, x):
        size = list(x.size()[:-1]) + [self.num_heads, self.depth]
        x = x.reshape(*size)
        # out shape = (B, *, H, S, DEP), * is additional dims
        return x.transpose(-2, -3)

    def forward(self, value, key, query, mask):
        # input.shape == (batch_size, vertices, d_model)
        # mask.shape == (batch_size, vertices, vertices)
        original_size = list(query.size())

        query = self.split_head(self.wq(query))
        key = self.split_head(self.wk(key))
        value = self.split_head(self.wv(value))


        old = mask.clone()
        scaled_attention, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        output = scaled_attention.transpose(-2, -3)

        size = original_size[:-1] + [self.hidden_size]
        concat_attention = scaled_attention.reshape(*size)

        output = self.dense(concat_attention) # to zero or not to zero

        # output.shape == (batch_size, vertices, d_model)
        # attention.shape == (batch_size, num_heads, vertices, vertices)
        return output, attention_weights
