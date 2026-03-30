"""Occipital lobe service consuming vision_raw and publishing vision_feat."""

from __future__ import annotations

import threading
import torch

from .occipital_lobe import OccipitalLobe
from .utils.message_bus import MessageBus
from .utils.config import load_config
from .utils.logger import get_logger


def main() -> None:
    cfg = load_config("configs/default.yaml")
    device = cfg["devices"].get("occipital_lobe", "cpu")

    logger = get_logger("occipital_service")
    bus = MessageBus()
    lobe = OccipitalLobe(device=device)

    def handle(arr: torch.Tensor | bytes) -> None:
        if isinstance(arr, bytes):
            tensor = torch.tensor([])
        else:
            tensor = torch.tensor(arr)
        tensor = tensor.to(device)
        feat = lobe.process(tensor)
        bus.publish_array("vision_feat", feat.cpu().numpy())

    bus.subscribe_array("vision_raw", handle)

    stop = threading.Event()
    try:
        stop.wait()
    except KeyboardInterrupt:
        logger.info("occipital service stopped")


if __name__ == "__main__":
    main()
