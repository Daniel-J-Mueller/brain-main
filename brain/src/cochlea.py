"""Cochlea service publishing audio embeddings via MessageBus."""

from __future__ import annotations

import numpy as np
import torch

from .sensors.cochlea import Cochlea
from .utils.audio_buffer import AudioBuffer
from .utils.message_bus import MessageBus
from .utils.config import load_config
from .utils.logger import get_logger


def main() -> None:
    cfg = load_config("configs/default.yaml")
    model_dir = cfg["models"]["whisper"]
    device = cfg["devices"].get("cochlea", "cpu")
    duration = float(cfg["settings"].get("audio_duration", 1.0))

    logger = get_logger("cochlea_service")
    bus = MessageBus()
    buffer = AudioBuffer(samplerate=16000, channels=1, buffer_seconds=duration * 2)
    cochlea = Cochlea(model_dir, device=device)

    try:
        while True:
            audio_np = buffer.read(duration)
            if len(audio_np) == 0:
                continue
            audio_tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
            emb = cochlea.encode([audio_tensor])
            bus.publish_array("audio_raw", emb.cpu().numpy())
    except KeyboardInterrupt:
        logger.info("cochlea service stopped")
    finally:
        buffer.close()


if __name__ == "__main__":
    main()
