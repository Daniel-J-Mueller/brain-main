"""Auditory cortex service consuming audio_raw and publishing audio_feat."""

from __future__ import annotations

import threading
import torch

from .auditory_cortex import AuditoryCortex
from .utils.message_bus import MessageBus
from .utils.config import load_config
from .utils.logger import get_logger


def main() -> None:
    cfg = load_config("configs/default.yaml")
    device = cfg["devices"].get("auditory_cortex", "cpu")

    logger = get_logger("auditory_service")
    bus = MessageBus()
    cortex = AuditoryCortex(device=device)

    def handle(arr: torch.Tensor | bytes) -> None:
        if isinstance(arr, bytes):
            tensor = torch.tensor([])
        else:
            tensor = torch.tensor(arr)
        tensor = tensor.to(device)
        feat = cortex.process(tensor)
        if feat.dim() == 3:
            feat = feat.mean(dim=1)
        bus.publish_array("audio_feat", feat.cpu().numpy())

    bus.subscribe_array("audio_raw", handle)

    stop = threading.Event()
    try:
        stop.wait()
    except KeyboardInterrupt:
        logger.info("auditory service stopped")


if __name__ == "__main__":
    main()
