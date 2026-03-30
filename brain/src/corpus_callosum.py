"""Cross-hemisphere communication bridge."""

from __future__ import annotations

import torch
from torch import nn
from pathlib import Path

from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class CorpusCallosum(nn.Module):
    """Placeholder pass-through layer approximating the corpus callosum."""

    def __init__(
        self,
        embed_dim: int = 768,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.bridge = nn.Identity()
        self.short_lora = FatigueLoRA(embed_dim, embed_dim, device=device)
        self.long_lora = LongTermLoRA(embed_dim, embed_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    @torch.no_grad()
    def transfer(self, embedding: torch.Tensor) -> torch.Tensor:
        """Relay ``embedding`` across hemispheres."""
        emb = embedding.to(self.device)
        adj = self.short_lora(emb) + self.long_lora(emb)
        return self.bridge(emb + adj)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
