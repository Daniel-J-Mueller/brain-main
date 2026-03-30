"""Project motor embeddings into interoceptive space."""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras
from .utils.sentinel import SentinelLinear


class InsularCortex(nn.Module):
    """Transforms motor cortex outputs before feedback."""

    def __init__(
        self,
        in_dim: int = 768,
        intero_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.proj = SentinelLinear(in_dim, intero_dim)
        self.short_lora = FatigueLoRA(in_dim, intero_dim, device=device)
        self.long_lora = LongTermLoRA(in_dim, intero_dim, device=device)
        self.act = nn.Tanh()
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    @torch.no_grad()
    def forward(self, motor_emb: torch.Tensor) -> torch.Tensor:
        motor_emb = motor_emb.to(self.device)
        base = self.proj(motor_emb)
        out = base + self.short_lora(motor_emb) + self.long_lora(motor_emb)
        return self.act(out)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
