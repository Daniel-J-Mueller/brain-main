"""Placeholder for autonomic control resembling the medulla oblongata."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn


class MedullaOblongata(nn.Module):
    """Output minimal breathing rhythm value."""

    def __init__(self, device: str = "cpu", persist_path: str | None = None) -> None:
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(0.5))
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def rhythm(self) -> float:
        return float(torch.sigmoid(self.rate).item())

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
