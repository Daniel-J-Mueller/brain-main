from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class Subiculum(nn.Module):
    """Relay hippocampal outputs back to cortex."""

    def __init__(
        self,
        hippo_dim: int = 768,
        cortex_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            SentinelLinear(hippo_dim, cortex_dim),
            nn.ReLU(),
        )
        self.short_lora = FatigueLoRA(hippo_dim, cortex_dim, device=device)
        self.long_lora = LongTermLoRA(hippo_dim, cortex_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Alias for :meth:`relay` so the module can be called directly."""
        return self.relay(memory)

    @torch.no_grad()
    def relay(self, memory: torch.Tensor) -> torch.Tensor:
        mem = memory.to(self.device)
        adj = self.short_lora(mem) + self.long_lora(mem)
        return self.proj(mem) + adj

    def save(self, path: str | None = None) -> None:
        target = Path(path) if path else self.persist_path
        if not target:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), target)
        save_loras(self, target)
