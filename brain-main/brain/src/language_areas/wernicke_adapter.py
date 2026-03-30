"""Adapter representing a semantic mapping step in Wernicke's area."""

import torch
from torch import nn
from pathlib import Path

from ..utils.sentinel import SentinelLinear


class WernickeAdapter(nn.Module):
    """Simple adapter that transforms token embeddings.

    The augmenter learns to map embeddings from the language areas
    into a richer space that better reflects recent context. It can be
    trained online via the :class:`Trainer` utilities.
    """

    def __init__(self, embed_dim: int = 768, hidden_dim: int = 1024, device: str = "cpu", persist_path: str | None = None) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(embed_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, embed_dim),
        )
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.to(self.device))

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
