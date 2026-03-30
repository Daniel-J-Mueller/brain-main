"""Associative token flow for coarse semantic sequencing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


class SemanticFlow:
    """Store probabilities of token transitions.

    Each token is referenced by its index from the token table.  Observed
    sequences update transition counts which are later normalized when
    retrieving probabilities or sampling the next token.
    """

    def __init__(self, vocab_size: int, persist_path: Optional[str] = None) -> None:
        self.vocab_size = vocab_size
        self.transitions: Dict[int, Dict[int, float]] = {}
        self.embeddings: List[np.ndarray] = []

        self.persist_base: Optional[Path] = None
        self.json_path: Optional[Path] = None
        self.emb_path: Optional[Path] = None
        if persist_path:
            base = Path(persist_path)
            # Treat provided path as base; append extensions for data files
            if base.suffix:
                base = base.with_suffix("")
            self.persist_base = base
            self.json_path = base.with_suffix(".json")
            self.emb_path = base.with_suffix(".npy")

        if self.json_path and self.json_path.exists():
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.transitions = {
                int(k): {int(k2): float(v2) for k2, v2 in v.items()} for k, v in data.items()
            }
        if self.emb_path and self.emb_path.exists():
            try:
                arr = np.load(self.emb_path)
                self.embeddings = [e.astype(np.float32) for e in arr]
            except Exception:
                self.embeddings = []

    def observe(self, token_ids: Iterable[int]) -> None:
        """Record transitions observed in ``token_ids``."""
        iterator = iter(token_ids)
        prev = next(iterator, None)
        for idx in iterator:
            if prev is None:
                prev = idx
                continue
            dest = self.transitions.setdefault(int(prev), {})
            dest[idx] = dest.get(idx, 0.0) + 1.0
            prev = idx

    def next_probabilities(self, idx: int) -> Dict[int, float]:
        """Return normalized probabilities of tokens following ``idx``."""
        dest = self.transitions.get(int(idx))
        if not dest:
            return {}
        total = float(sum(dest.values()))
        if total <= 0.0:
            return {}
        return {i: count / total for i, count in dest.items()}

    def sample_next(self, idx: int, temperature: float = 1.0) -> int | None:
        """Sample the next token index after ``idx``."""
        probs = self.next_probabilities(idx)
        if not probs:
            return None
        tokens = list(probs.keys())
        weights = np.array(list(probs.values()), dtype=np.float64)
        if temperature != 1.0:
            weights = weights ** (1.0 / max(temperature, 1e-5))
            weights = weights / weights.sum()
        choice = int(np.random.choice(tokens, p=weights))
        return choice

    def save(self, path: str | None = None) -> None:
        """Persist the transition table to ``path`` or ``persist_path``."""
        base = Path(path).with_suffix("") if path else self.persist_base
        if not base:
            return
        base.parent.mkdir(parents=True, exist_ok=True)
        with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.transitions, f)
        np.save(base.with_suffix(".npy"), np.array(self.embeddings, dtype=np.float32))

    # ------------------------------------------------------------------
    # Embedding-based cause/effect modelling
    # ------------------------------------------------------------------

    def _get_index(self, emb: np.ndarray, sim_threshold: float = 0.95) -> int:
        """Return index of ``emb`` in the table, adding it if new."""
        if len(self.embeddings) == 0:
            self.embeddings.append(emb.astype(np.float32))
            return 0

        vec = emb.astype(np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(vec) + 1e-8
        sims = np.dot(self.embeddings, vec) / norms
        best = int(np.argmax(sims))
        if sims[best] >= sim_threshold:
            self.embeddings[best] = (self.embeddings[best] + vec) / 2.0
            return best

        self.embeddings.append(vec)
        return len(self.embeddings) - 1

    def observe_transition(
        self,
        cause: np.ndarray,
        effect: np.ndarray,
        diff_threshold: float = 0.1,
        sim_threshold: float = 0.95,
    ) -> None:
        """Record a cause → effect embedding transition."""
        if cause.ndim > 1:
            cause = cause.reshape(-1)
        if effect.ndim > 1:
            effect = effect.reshape(-1)

        diff = 1.0 - float(
            np.dot(cause, effect)
            / (np.linalg.norm(cause) * np.linalg.norm(effect) + 1e-8)
        )
        if diff < diff_threshold:
            return

        c_idx = self._get_index(cause, sim_threshold)
        e_idx = self._get_index(effect, sim_threshold)
        dest = self.transitions.setdefault(int(c_idx), {})
        dest[e_idx] = dest.get(e_idx, 0.0) + 1.0

    def predict_next_embedding(
        self, emb: np.ndarray, temperature: float = 1.0, sim_threshold: float = 0.95
    ) -> Optional[np.ndarray]:
        """Return a sampled effect embedding predicted from ``emb``."""
        if emb.ndim > 1:
            emb = emb.reshape(-1)
        idx = self._get_index(emb, sim_threshold)
        probs = self.next_probabilities(idx)
        if not probs:
            return None
        keys = list(probs.keys())
        weights = np.array(list(probs.values()), dtype=np.float64)
        if temperature != 1.0:
            weights = weights ** (1.0 / max(temperature, 1e-5))
            weights = weights / weights.sum()
        choice = int(np.random.choice(keys, p=weights))
        if choice >= len(self.embeddings):
            return None
        return self.embeddings[choice]
