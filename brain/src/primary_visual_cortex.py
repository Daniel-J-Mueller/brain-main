"""Edge and orientation detector approximating the primary visual cortex."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class PrimaryVisualCortex(nn.Module):
    """Simplistic feature extractor for early visual processing."""

    def __init__(
        self,
        input_dim: int = 512,
        output_dim: int = 512,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, output_dim),
            nn.ReLU(),
        )
        self.short_lora = FatigueLoRA(input_dim, output_dim, device=device)
        self.long_lora = LongTermLoRA(input_dim, output_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def extract(self, emb: torch.Tensor) -> torch.Tensor:
        emb = emb.to(self.device)
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.net(emb) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
