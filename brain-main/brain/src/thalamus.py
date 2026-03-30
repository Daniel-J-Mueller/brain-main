"""Central relay and gate for sensory information."""

from queue import Queue, Full, Empty
from typing import Any, Dict


class Thalamus:
    """Simplified thalamic relay with drop-when-busy semantics."""

    def __init__(self) -> None:
        # maxsize=1 implements "inattentional deafness" when overwhelmed
        self.queues: Dict[str, Queue] = {
            "vision": Queue(maxsize=1),
            "audio": Queue(maxsize=1),
            "intero": Queue(maxsize=1),
            "motor": Queue(maxsize=1),
        }
        self.arousal: float = 1.0  # 0.0 = asleep, 1.0 = fully alert

    def submit(self, modality: str, sample: Any) -> None:
        """Submit a sensory sample for relay.

        If the queue is full, the sample is silently dropped.
        """
        try:
            self.queues[modality].put(sample, block=False)
        except Full:
            # Overflow → drop the sample
            pass

    def relay(self, modality: str) -> Any:
        """Retrieve the latest sample for ``modality`` if arousal permits."""
        if self.arousal <= 0.0:
            # Gate closed
            return None
        try:
            return self.queues[modality].get(block=False)
        except Empty:
            return None

    def set_arousal(self, level: float) -> None:
        """Adjust arousal level (0.0–1.0)."""
        self.arousal = max(0.0, min(1.0, level))
