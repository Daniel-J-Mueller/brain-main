"""Track novelty of produced tokens for intrinsic motivation."""

from __future__ import annotations

from typing import Dict, Any
import torch


class CuriosityTracker:
    """Count token usage and provide novelty bonuses."""

    def __init__(self) -> None:
        self.counts: Dict[int, int] = {}
        self.last_token: int | None = None
        self.repeat_streak: int = 0

    def bonus(self, token_id: int) -> float:
        """Return intrinsic bonus for ``token_id`` based on how rarely it was used."""
        count = self.counts.get(token_id, 0)
        return 1.0 / (1.0 + float(count))

    def update(self, token_id: int) -> None:
        """Record another occurrence of ``token_id``."""
        if token_id == self.last_token:
            self.repeat_streak += 1
        else:
            self.repeat_streak = 1
        self.last_token = token_id
        self.counts[token_id] = self.counts.get(token_id, 0) + 1

    def transform(self, emb: torch.Tensor) -> torch.Tensor:
        """Jitter ``emb`` when the same token repeats."""
        if self.repeat_streak < 2:
            return emb

        strength = min(1.0, self.repeat_streak / 2.0)
        # Additive jitter instead of multiplicative so zero embeddings can
        # still diverge.  Frequency noise preserves overall structure while
        # providing stochastic exploration.
        freq = torch.fft.rfft(emb, dim=-1)
        noise = (torch.rand_like(freq.real) - 0.5) * 2.0 * strength
        freq = freq + noise + 0j
        return torch.fft.irfft(freq, n=emb.shape[-1], dim=-1)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "counts": dict(self.counts),
            "last_token": self.last_token,
            "repeat_streak": self.repeat_streak,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.counts = dict(state.get("counts", {}))
        self.last_token = state.get("last_token")
        self.repeat_streak = int(state.get("repeat_streak", 0))
