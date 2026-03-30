from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class GlobusPallidus(nn.Module):
    """Braking signal approximating the globus pallidus."""

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
            nn.Sigmoid(),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def brake(self, context: torch.Tensor) -> float:
        ctx = context.to(self.device)
        out = self.net(ctx) + self.short_lora(ctx) + self.long_lora(ctx)
        return float(out.squeeze())

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
