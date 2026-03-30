"""Emotional valence scoring approximating the amygdala."""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras
from .utils.sentinel import SentinelLinear


class Amygdala(nn.Module):
    """Assign a valence score to context embeddings."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 64,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.act = nn.Tanh()
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    @torch.no_grad()
    def evaluate(self, embedding: torch.Tensor) -> float:
        """Return valence in ``[-1, 1]`` for ``embedding``."""
        emb = embedding.to(self.device)
        val = self.net(emb) + self.short_lora(emb) + self.long_lora(emb)
        val = self.act(val)
        return float(val.squeeze())

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
