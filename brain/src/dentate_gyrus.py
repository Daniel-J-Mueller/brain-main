from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class DentateGyrus(nn.Module):
    """Encode new memories approximating the dentate gyrus."""

    def __init__(
        self,
        cortex_dim: int = 768,
        hippo_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            SentinelLinear(cortex_dim, hippo_dim),
            nn.ReLU(),
        )
        self.short_lora = FatigueLoRA(cortex_dim, hippo_dim, device=device)
        self.long_lora = LongTermLoRA(cortex_dim, hippo_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def encode(self, embedding: torch.Tensor) -> torch.Tensor:
        emb = embedding.to(self.device)
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.proj(emb) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
