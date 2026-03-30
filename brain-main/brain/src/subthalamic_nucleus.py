"""Decision inertia mechanism delaying impulsive actions."""

from __future__ import annotations

import torch
from torch import nn

from .utils.sentinel import SentinelLinear


class SubthalamicNucleus(nn.Module):
    """Predict inhibitory factor based on current context.

    The nucleus outputs a probability in ``[0,1]`` representing how much
    to inhibit the currently forming motor plan.  A small reinforcement
    rule adapts ``threshold`` based on feedback from the amygdala so the
    model becomes more or less conservative over time.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 64, device: str = "cpu") -> None:
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.device = device
        # threshold starts neutral and is nudged by ``reinforce`` calls
        self.threshold = 0.5
        self.baseline = 0.0
        self.to(device)

    @torch.no_grad()
    def inhibition(self, context: torch.Tensor) -> float:
        """Return a value in ``[0,1]`` describing how much to slow actions."""
        ctx = context.to(self.device)
        level = float(self.net(ctx))
        # return inhibition adjusted by the current threshold
        return max(0.0, min(1.0, level - self.threshold))

    @torch.no_grad()
    def reinforce(self, reward: float, lr: float = 0.05, beta: float = 0.1) -> None:
        """Update ``threshold`` using moving-average baseline of ``reward``."""
        self.baseline = (1 - beta) * self.baseline + beta * reward
        adj = reward - self.baseline
        self.threshold -= lr * adj
        self.threshold = max(0.0, min(1.0, float(self.threshold)))
