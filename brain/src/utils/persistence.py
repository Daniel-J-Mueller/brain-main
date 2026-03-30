"""Helper functions for msgpack-based persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import msgpack
import msgpack_numpy
import numpy as np


def save_msgpack(path: str | Path, obj: Any) -> None:
    """Serialize ``obj`` to ``path`` using msgpack."""
    data = msgpack.dumps(obj, default=msgpack_numpy.encode)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def load_msgpack(path: str | Path) -> Any:
    """Load an object previously written by :func:`save_msgpack`."""
    with open(path, "rb") as f:
        data = f.read()
    return msgpack.loads(data, object_hook=msgpack_numpy.decode)
