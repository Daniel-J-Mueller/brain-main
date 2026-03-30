"""Simplistic episodic memory with cross-modal associations."""

from __future__ import annotations

from typing import Dict, List, Optional

from pathlib import Path

import numpy as np
import psutil

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    faiss = None


class Hippocampus:
    """Episodic memory storing embeddings for multiple modalities."""

    def __init__(
        self,
        dims: Dict[str, int],
        capacity: int = 1000,
        persist_path: Optional[str] = None,
        use_faiss: bool = True,
        compressed: bool = True,
        recall_threshold: float = 0.0,
        *,
        salience_threshold: float = 0.0,
    ) -> None:
        self.dims = dims
        self.capacity = capacity
        # Each entry is a mapping ``modality -> embedding`` plus optional ``valence``
        self.memory: List[Dict[str, np.ndarray | float]] = []
        self.persist_path = Path(persist_path) if persist_path else None
        self.compressed = compressed
        self.use_faiss = use_faiss and faiss is not None
        self.index: Dict[str, "faiss.Index"] = {}
        self.mapping: Dict[str, List[int]] = {}
        self.recall_threshold = recall_threshold
        self.salience_threshold = salience_threshold
        self.process = psutil.Process()

        if self.persist_path and self.persist_path.exists():
            try:
                if self.persist_path.suffix == ".npz":
                    data = np.load(self.persist_path, allow_pickle=True)
                    self.memory = data["memory"].tolist()
                else:
                    self.memory = np.load(self.persist_path, allow_pickle=True).tolist()
            except Exception:
                self.memory = []

        if self.use_faiss:
            self._rebuild_index()

    def memory_usage_gb(self) -> float:
        """Return approximate process memory usage in gigabytes."""
        mem = self.process.memory_info().rss
        return mem / 1e9

    def _rebuild_index(self) -> None:
        """Recreate FAISS indices from current memory."""
        if not self.use_faiss:
            return
        self.index = {}
        self.mapping = {}
        for modality, dim in self.dims.items():
            self.index[modality] = faiss.IndexFlatIP(dim)
            self.mapping[modality] = []
        for i, ep in enumerate(self.memory):
            for modality, dim in self.dims.items():
                if modality not in ep:
                    continue
                vec = ep[modality]
                if vec.ndim > 1:
                    vec = vec.mean(axis=0)
                self.index[modality].add(vec.reshape(1, -1).astype("float32"))
                self.mapping[modality].append(i)

    def add_episode(
        self,
        episode: Dict[str, np.ndarray],
        valence: float = 0.0,
        *,
        salience: float = 1.0,
    ) -> None:
        """Store a set of embeddings for different modalities with a valence tag."""

        if salience < self.salience_threshold:
            return

        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        # Ensure all arrays are float32 for consistency
        clean = {}
        for m, emb in episode.items():
            if m in self.dims and emb.shape[-1] != self.dims[m]:
                continue
            clean[m] = emb.astype(np.float32)
        clean["valence"] = float(valence)
        self.memory.append(clean)
        if self.use_faiss:
            self._rebuild_index()

    def query(self, modality: str, embedding: np.ndarray, k: int = 5) -> Dict[str, np.ndarray]:
        """Retrieve averaged embeddings from the closest episodes.

        Parameters
        ----------
        modality:
            The key used to compare against stored episodes.
        embedding:
            The query embedding of the same modality.
        """

        if not self.memory:
            return {}
        emb = embedding.astype(np.float32)
        best_score = 0.0
        if self.use_faiss and modality in self.index and self.index[modality].ntotal > 0:
            scores, faiss_idx = self.index[modality].search(
                emb.reshape(1, -1), min(k, self.index[modality].ntotal)
            )
            idx = [self.mapping[modality][i] for i in faiss_idx[0]]
            best_score = float(scores[0][0]) if scores.size > 0 else 0.0
        else:
            scores = []
            for ep in self.memory:
                if modality not in ep:
                    scores.append(-1.0)
                    continue
                m = ep[modality]
                if m.ndim > 1:
                    m = m.mean(axis=0)
                if m.shape[0] != emb.shape[0]:
                    scores.append(-1.0)
                    continue
                score = float(
                    np.dot(emb, m)
                    / (np.linalg.norm(emb) * np.linalg.norm(m) + 1e-8)
                )
                scores.append(score)
            idx = np.argsort(scores)[-k:][::-1]
            if scores:
                best_score = float(scores[idx[0]])
        if best_score < self.recall_threshold:
            return {}
        collected: Dict[str, List[np.ndarray]] = {m: [] for m in self.dims}
        valences: List[float] = []
        for i in idx:
            ep = self.memory[i]
            for m, val in ep.items():
                if m == "valence":
                    valences.append(float(val))
                    continue
                if val.ndim > 1:
                    val = val.mean(axis=0)
                collected.setdefault(m, []).append(val)

        result = {
            m: np.mean(vals, axis=0) for m, vals in collected.items() if len(vals) > 0
        }
        if valences:
            result["valence"] = float(np.mean(valences))
        return result

    def decay(self, rate: float = 0.99) -> None:
        """Gradually weaken all stored embeddings."""
        for ep in self.memory:
            for m, val in ep.items():
                if m == "valence":
                    ep[m] = float(val) * rate
                else:
                    ep[m] = val * rate
        if self.use_faiss:
            self._rebuild_index()

    def clear(self) -> None:
        """Remove all stored episodes."""
        self.memory.clear()
        if self.use_faiss:
            self._rebuild_index()

    def save(self) -> None:
        """Persist memory to disk if ``persist_path`` is set."""
        if not self.persist_path:
            return
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.array(self.memory, dtype=object)
        if self.compressed or self.persist_path.suffix == ".npz":
            np.savez_compressed(self.persist_path, memory=arr)
        else:
            np.save(self.persist_path, arr, allow_pickle=True)


class DistributedHippocampus:
    """Wrap multiple :class:`Hippocampus` shards for larger capacity."""

    def __init__(
        self,
        dims: Dict[str, int],
        num_shards: int = 1,
        *,
        shard_paths: Optional[List[str]] = None,
        independent: bool = False,
        **kwargs,
    ) -> None:
        if shard_paths is None:
            shard_paths = [None] * num_shards
        if len(shard_paths) < num_shards:
            shard_paths = shard_paths + [None] * (num_shards - len(shard_paths))
        self.shards = [
            Hippocampus(dims, persist_path=path, **kwargs) for path in shard_paths
        ]
        self.independent = independent
        self.next_shard = 0

    def add_episode(
        self, episode: Dict[str, np.ndarray], valence: float = 0.0, *, salience: float = 1.0
    ) -> None:
        if self.independent:
            target = self.shards[self.next_shard]
            target.add_episode(episode, valence, salience=salience)
            self.next_shard = (self.next_shard + 1) % len(self.shards)
        else:
            for shard in self.shards:
                shard.add_episode(episode, valence, salience=salience)

    def query(self, modality: str, embedding: np.ndarray, k: int = 5) -> Dict[str, np.ndarray]:
        merged: Dict[str, List[np.ndarray]] = {}
        for shard in self.shards:
            res = shard.query(modality, embedding, k)
            for key, val in res.items():
                merged.setdefault(key, []).append(val)
        if not merged:
            return {}
        return {
            k: np.mean(v, axis=0) if isinstance(v[0], np.ndarray) else float(np.mean(v))
            for k, v in merged.items()
        }

    def decay(self, rate: float = 0.99) -> None:
        for shard in self.shards:
            shard.decay(rate)

    def memory_usage_gb(self) -> float:
        """Return combined memory usage of all shards."""
        return float(sum(shard.memory_usage_gb() for shard in self.shards))

    def clear(self) -> None:
        for shard in self.shards:
            shard.clear()

    def save(self) -> None:
        for shard in self.shards:
            shard.save()
