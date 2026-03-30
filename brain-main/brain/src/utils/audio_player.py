"""Thread-safe audio playback utility using ``sounddevice``."""

from __future__ import annotations

import threading
from typing import Iterable, Optional
import queue
import numpy as np
import sounddevice as sd


_queue: "queue.Queue[tuple[np.ndarray, int]]" = queue.Queue()
_worker: Optional[threading.Thread] = None
_lock = threading.Lock()


def _playback_loop() -> None:
    """Internal worker that plays audio sequentially."""

    while True:
        arr, rate = _queue.get()
        if arr.size == 0:
            continue
        try:
            sd.play(arr, samplerate=rate)
            sd.wait()
        except Exception as exc:  # pragma: no cover - device errors
            print(f"[AudioPlayer] {exc}")


def play_audio(audio: Iterable[float] | np.ndarray, samplerate: int = 16000) -> None:
    """Queue ``audio`` for playback in a background thread."""

    arr = np.array(list(audio), dtype=np.float32).flatten()
    if arr.size == 0:
        return

    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_playback_loop, daemon=True)
            _worker.start()
    _queue.put((arr, samplerate))
