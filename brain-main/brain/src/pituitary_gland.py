"""Hormone relay approximating the pituitary gland."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class PituitaryGland(nn.Module):
    """Convert hypothalamus signals into peripheral hormones."""

    def __init__(
        self,
        input_dim: int = 4,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.short_lora = FatigueLoRA(input_dim, input_dim, device=device)
        self.long_lora = LongTermLoRA(input_dim, input_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def release(self, levels: torch.Tensor) -> torch.Tensor:
        lv = levels.to(self.device)
        adj = self.short_lora(lv) + self.long_lora(lv)
        return self.net(lv) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
