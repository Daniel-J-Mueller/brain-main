"""Motor plan preparer approximating the premotor cortex."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class PremotorCortex(nn.Module):
    """Transform context embeddings into preliminary motor plans."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, input_dim),
        )
        self.short_lora = FatigueLoRA(input_dim, input_dim, device=device)
        self.long_lora = LongTermLoRA(input_dim, input_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def prepare(self, context: torch.Tensor) -> torch.Tensor:
        ctx = context.to(self.device)
        adj = self.short_lora(ctx) + self.long_lora(ctx)
        return self.net(ctx) + adj

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
