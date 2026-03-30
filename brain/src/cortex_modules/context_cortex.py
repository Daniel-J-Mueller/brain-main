"""Temporal context encoder."""

import torch
from torch import nn


class ContextCortex(nn.Module):
    """GRU-based context representation."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024, device: str = "cpu") -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.hidden = torch.zeros(1, 1, hidden_dim, device=device)
        self.device = device
        self.to(device)

    @torch.no_grad()
    def step(self, embedding: torch.Tensor) -> torch.Tensor:
        out, self.hidden = self.gru(embedding.unsqueeze(0).to(self.device), self.hidden)
        return out.squeeze(0)
