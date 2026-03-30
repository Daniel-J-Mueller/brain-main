"""Bridge between hippocampus and cortex."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class EntorhinalCortex(nn.Module):
    """Map cortical embeddings to hippocampal space and back."""

    def __init__(
        self,
        embed_dim: int = 768,
        hippo_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.to_hippo = nn.Sequential(
            SentinelLinear(embed_dim, hippo_dim),
            nn.ReLU(),
        )
        self.from_hippo = nn.Sequential(
            SentinelLinear(hippo_dim, embed_dim),
            nn.ReLU(),
        )
        self.short_lora = FatigueLoRA(embed_dim, hippo_dim, device=device)
        self.long_lora = LongTermLoRA(embed_dim, hippo_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def funnel(self, embedding: torch.Tensor) -> torch.Tensor:
        emb = embedding.to(self.device)
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.to_hippo(emb) + adj

    @torch.no_grad()
    def project(self, memory: torch.Tensor) -> torch.Tensor:
        mem = memory.to(self.device)
        return self.from_hippo(mem)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
