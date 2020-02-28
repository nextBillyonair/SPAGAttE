import torch
from torch.nn import Module, Linear


class PairwiseBilinear(Module):
    def __init__(self, input_size=64, bias=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = input_size
        self.linear = Linear(input_size, input_size, bias=bias)

    def forward(self, embeddings):
        dot_prod = torch.bmm(self.linear(embeddings), embeddings.transpose(-1, -2))
        return dot_prod


class PairwiseDot(Module):
    def forward(self, embeddings):
        dot_prod = torch.bmm(embeddings, embeddings.transpose(-1, -2))
        return dot_prod


class PairwiseDistance(Module):
    def forward(self, embeddings, p=2):
        n = embeddings.size(-2)
        d = embeddings.size(-1)
        x = embeddings.expand(n, n, d)
        y = embeddings.expand(n, n, d).transpose(-2, -3)
        dist = torch.pow(x - y, p).sum(-1).unsqueeze(0)
        return dist
