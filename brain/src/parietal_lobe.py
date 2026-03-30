"""Spatial attention module approximating the parietal lobe."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class ParietalLobe(nn.Module):
    """Combine visual features with proprioceptive context."""

    def __init__(
        self,
        vision_dim: int = 128,
        output_dim: int = 128,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(vision_dim, output_dim),
            nn.ReLU(),
            SentinelLinear(output_dim, output_dim),
        )
        self.short_lora = FatigueLoRA(vision_dim, output_dim, device=device)
        self.long_lora = LongTermLoRA(vision_dim, output_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def attend(self, vision_feat: torch.Tensor) -> torch.Tensor:
        feat = vision_feat.to(self.device)
        adj = self.short_lora(feat) + self.long_lora(feat)
        return self.net(feat) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
