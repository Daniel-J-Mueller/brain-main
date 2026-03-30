"""High-level planner aggregating frontal lobe subregions."""

from __future__ import annotations

from pathlib import Path
import torch
from torch import nn

from .prefrontal_cortex import PrefrontalCortex
from .orbitofrontal_cortex import OrbitofrontalCortex
from .premotor_cortex import PremotorCortex
from .middle_frontal_gyrus import MiddleFrontalGyrus
from .inferior_frontal_gyrus import InferiorFrontalGyrus


class FrontalLobe(nn.Module):
    """Coordinate decision making and motor preparation."""

    def __init__(
        self,
        device: str = "cpu",
        persist_path: str | None = None,
        *,
        ifg_feedback_buffer: float = 30.0,
    ) -> None:
        super().__init__()
        self.prefrontal = PrefrontalCortex(device=device)
        self.orbitofrontal = OrbitofrontalCortex(device=device)
        self.premotor = PremotorCortex(device=device)
        self.middle_frontal = MiddleFrontalGyrus(device=device)
        self.inferior_frontal = InferiorFrontalGyrus(
            device=device, feedback_buffer=ifg_feedback_buffer
        )
        self.device = device
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def bias(self, context: torch.Tensor) -> float:
        pf = self.prefrontal(context)
        ofc = self.orbitofrontal.assess(context)
        inhibit = self.inferior_frontal.inhibit(context)
        w = self.middle_frontal.weight(context)
        score = float(pf.squeeze()) * (1.0 - inhibit) * w * (0.5 + 0.5 * ofc)
        return max(0.0, min(1.0, score))

    @torch.no_grad()
    def plan(self, context: torch.Tensor) -> torch.Tensor:
        return self.premotor.prepare(context)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
