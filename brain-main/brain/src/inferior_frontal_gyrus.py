"""Response inhibition approximating the inferior frontal gyrus."""

from __future__ import annotations

from pathlib import Path
import time
from collections import deque
import torch
from torch import nn

from .utils.sentinel import SentinelLinear
from .utils.adapters import FatigueLoRA, LongTermLoRA, save_loras


class InferiorFrontalGyrus(nn.Module):
    """Predict suppression strength for ongoing behaviour."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 64,
        device: str = "cpu",
        persist_path: str | None = None,
        feedback_buffer: float = 30.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.short_lora = FatigueLoRA(input_dim, 1, device=device)
        self.long_lora = LongTermLoRA(input_dim, 1, device=device)
        self.device = device
        self.to(device)
        self.feedback_buffer = float(feedback_buffer)
        self._recent: deque[tuple[float, torch.Tensor]] = deque()
        from .trainer import Trainer

        self._trainer = Trainer()
        self.persist_path = Path(persist_path) if persist_path else None
        if self.persist_path and self.persist_path.exists():
            self.load_state_dict(torch.load(self.persist_path, map_location=device))

    @torch.no_grad()
    def inhibit(self, context: torch.Tensor) -> float:
        ctx = context.to(self.device)
        out = self.net(ctx) + self.short_lora(ctx) + self.long_lora(ctx)
        return float(out.squeeze())

    def _trim_recent(self) -> None:
        cutoff = time.time() - self.feedback_buffer
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

    def record_output(self, context: torch.Tensor) -> None:
        """Store ``context`` for potential feedback."""
        self._recent.append((time.time(), context.detach().cpu()))
        self._trim_recent()

    @torch.no_grad()
    def reinforce(self, rating: int) -> None:
        """Adjust inhibition based on user ``rating``.

        Negative ratings increase suppression for recent contexts while
        positive ratings reduce it.  Associations fade after
        ``feedback_buffer`` seconds.
        """

        if rating == 0:
            return
        self._trim_recent()
        if not self._recent:
            return
        target_val = 1.0 if rating < 0 else 0.0
        target = torch.tensor([[target_val]], device=self.device)
        scale = abs(float(rating)) / 5.0
        modules = [self.net, self.short_lora, self.long_lora]
        now = time.time()
        for ts, ctx in self._recent:
            age = now - ts
            weight = max(0.0, 1.0 - age / self.feedback_buffer)
            if weight <= 0.0:
                continue
            ctx = ctx.to(self.device).unsqueeze(0)
            actual = self.net(ctx) + self.short_lora(ctx) + self.long_lora(ctx)
            self._trainer.align(modules, target, actual, lr_scale=scale * weight)

    def save(self, path: str | None = None) -> None:
        target = path or self.persist_path
        if not target:
            return
        torch.save(self.state_dict(), target)
        save_loras(self, target)
