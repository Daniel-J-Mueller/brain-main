from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class Precuneus(nn.Module):
    """Imagination and self-reflection placeholder."""

    def __init__(
        self,
        input_dim: int = 768,
        output_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, output_dim),
            nn.ReLU(),
            SentinelLinear(output_dim, output_dim),
        )
        self.short_lora = FatigueLoRA(input_dim, output_dim, device=device)
        self.long_lora = LongTermLoRA(input_dim, output_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def reflect(self, embedding: torch.Tensor) -> torch.Tensor:
        emb = embedding.to(self.device)
        expected = self.short_lora.A.size(1)
        if emb.size(-1) != expected:
            if emb.size(-1) < expected:
                pad = expected - emb.size(-1)
                emb = torch.nn.functional.pad(emb, (0, pad))
            else:
                emb = emb[..., :expected]
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.net(emb) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
