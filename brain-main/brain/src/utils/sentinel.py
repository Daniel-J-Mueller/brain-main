SENTINEL = -1e9
UNTRAINED_INIT = 1e-3

import torch
from torch import nn
from torch.nn import functional as F

class SentinelLinear(nn.Linear):
    """Linear layer that treats SENTINEL weights as disabled."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features, out_features, bias=bias)
        nn.init.constant_(self.weight, SENTINEL)
        if self.bias is not None:
            nn.init.constant_(self.bias, SENTINEL)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = torch.where(self.weight == SENTINEL, 0.0, self.weight)
        bias = self.bias
        if bias is not None:
            bias = torch.where(bias == SENTINEL, 0.0, bias)
        return F.linear(input, weight, bias)
