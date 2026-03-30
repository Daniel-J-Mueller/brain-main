"""Retina service publishing visual embeddings via MessageBus."""

from __future__ import annotations

import cv2
from PIL import Image

from .sensors.retina import Retina
from .utils.message_bus import MessageBus
from .utils.camera import Camera
from .utils.config import load_config
from .utils.logger import get_logger


def main() -> None:
    cfg = load_config("configs/default.yaml")
    model_dir = cfg["models"]["clip"]
    device = cfg["devices"].get("retina", "cpu")

    logger = get_logger("retina_service")
    bus = MessageBus()
    cam = Camera()
    retina = Retina(model_dir, device=device)

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb).resize((224, 224))
            emb = retina.encode([img])
            bus.publish_array("vision_raw", emb.cpu().numpy())
    except KeyboardInterrupt:
        logger.info("retina service stopped")
    finally:
        cam.release()


if __name__ == "__main__":
    main()
