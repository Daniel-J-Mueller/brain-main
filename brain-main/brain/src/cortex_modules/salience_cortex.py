"""Novelty and importance detector."""

import torch
from torch import nn

from ..utils.sentinel import SentinelLinear


class SalienceCortex(nn.Module):
    """Compute a salience score for fused embeddings."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def score(self, embedding: torch.Tensor) -> float:
        val = self.net(embedding.to(self.device))
        return float(val)
