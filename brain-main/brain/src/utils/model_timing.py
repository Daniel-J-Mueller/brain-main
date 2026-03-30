from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

@contextmanager
def log_timing(model: str, operation: str, enabled: bool, log_dir: Path) -> None:
    """Log time spent performing a model operation.

    Parameters
    ----------
    model:
        Name of the model being executed.
    operation:
        Either ``"inference"`` or ``"training"``.
    enabled:
        When ``False`` the context manager does nothing.
    log_dir:
        Directory in which to append ``model_timing_debug.log``.
    """
    if not enabled:
        yield
        return
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "model_timing_debug.log"
        duration = end - start
        with log_path.open("a") as f:
            f.write(f"{start},{model},{operation},{duration}\n")
