"""Adaptive gating combining frontal signals approximating the supplementary motor area."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class SupplementaryMotorArea(nn.Module):
    """Mix premotor, prefrontal and IFG signals to yield a dynamic threshold."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 64,
        device: str = "cpu",
        ramp_duration: float = 300.0,
        target_threshold: float = 0.75,
        use_ramping: bool = False,
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim + 2, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.short_lora = FatigueLoRA(input_dim + 2, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim + 2, 1, device=device)
        self.device = device
        self.to(device)
        self.start_time = time.time()
        self.ramp_duration = float(ramp_duration)
        self.target_threshold = float(target_threshold)
        self.use_ramping = use_ramping
        self.min_thresh = 0.25
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def compute_threshold(
        self,
        premotor_emb: torch.Tensor,
        pfc_val: float,
        ifg_val: float,
        dopamine: float = 0.5,
    ) -> float:
        inputs = torch.cat(
            [premotor_emb.to(self.device), torch.tensor([[pfc_val, ifg_val]], device=self.device)],
            dim=1,
        )
        out = self.net(inputs) + self.short_lora(inputs) + self.long_lora(inputs)
        mix = float(out.squeeze())
        if self.use_ramping:
            ramp = min(1.0, (time.time() - self.start_time) / self.ramp_duration)
            base = self.min_thresh + (self.target_threshold - self.min_thresh) * ramp
        else:
            base = self.target_threshold
        base -= 0.1 * (dopamine - 0.5)
        thresh = (base + mix) / 2.0
        return max(0.0, min(1.0, thresh))

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
