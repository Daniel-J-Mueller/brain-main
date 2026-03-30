"""Speculative output buffer representing the temporal lobe."""

from __future__ import annotations

from typing import List

import torch

from .language_areas.wernickes_area import WernickesArea


class TemporalLobe:
    """Neurosymbolic region storing speculative motor tokens."""

    def __init__(self, max_tokens: int = 32) -> None:
        self.max_tokens = max_tokens
        self._buffer: List[str] = []

    def add_speculation(self, tokens: List[str]) -> None:
        """Append speculative tokens to the buffer."""
        for tok in tokens:
            if len(self._buffer) >= self.max_tokens:
                self._buffer.pop(0)
            self._buffer.append(tok)

    def consume(self, token: str) -> None:
        """Remove ``token`` from the buffer if present."""
        if self._buffer:
            if self._buffer[0] == token:
                self._buffer.pop(0)
                return
            try:
                self._buffer.remove(token)
            except ValueError:
                pass

    def embedding(self, wernicke: WernickesArea) -> torch.Tensor:
        """Return combined embedding of remaining speculative tokens."""
        if not self._buffer:
            emb_dim = getattr(
                wernicke.model.config,
                "n_embd",
                getattr(wernicke.model.config, "hidden_size", 768),
            )
            return torch.zeros(1, emb_dim, device=wernicke.device)
        text = "".join(self._buffer)
        return wernicke.encode([text]).mean(dim=1)

    def clear(self) -> None:
        self._buffer.clear()
