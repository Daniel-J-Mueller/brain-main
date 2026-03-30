"""Motor error correction approximating the cerebellum."""

import torch
from torch import nn
from pathlib import Path

from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras
from .utils.sentinel import SentinelLinear


class Cerebellum(nn.Module):
    """Predict corrective adjustments for motor embeddings."""

    def __init__(
        self,
        vision_dim: int = 128,
        motor_dim: int = 768,
        hidden_dim: int = 256,
        device: str = "cpu",
        persist_path: str | None = None,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(vision_dim + motor_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, motor_dim),
        )
        self.short_lora = FatigueLoRA(vision_dim + motor_dim, motor_dim, device=device)
        self.long_lora = LongTermLoRA(vision_dim + motor_dim, motor_dim, device=device)
        self.device = device
        self.to(device)
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            state = torch.load(self.persist_path, map_location=device)
            self.load_state_dict(state)

    @torch.no_grad()
    def adjust(self, motor_emb: torch.Tensor, vision_feat: torch.Tensor) -> torch.Tensor:
        """Return adjusted motor embedding using visual feedback."""
        m = motor_emb
        v = vision_feat
        if m.dim() == 3:
            m_base = m.mean(dim=1)
        else:
            m_base = m
        if v.dim() == 3:
            v_base = v.mean(dim=1)
        else:
            v_base = v
        inp = torch.cat([m_base.to(self.device), v_base.to(self.device)], dim=-1)
        correction = self.net(inp) + self.short_lora(inp) + self.long_lora(inp)
        if m.dim() == 3:
            correction = correction.unsqueeze(1).expand_as(m)
        return m.to(self.device) + correction

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
