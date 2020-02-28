import torch
from torch.nn import Module, Linear

# Lernable Residual
class Residual(Module):

    def __init__(self, input_size, hidden_size=128):
        super().__init__()
        self.wa = Linear(2 * input_size, hidden_size)
        self.va = Linear(hidden_size, 1)

    def forward(self, old, new, return_weights=True):
        # old == new == (B, *, S, E)
        combine = torch.cat((old, new), dim=-1)
        weights = torch.sigmoid(self.va(torch.tanh(self.wa(combine))))
        attended = weights * old + (1 - weights) * new
        # out == (B, *, S, E), weight == (B, *, S, 2)
        if return_weights:
            return attended, torch.cat((weights, (1 - weights)), dim=-1)
        return attended
