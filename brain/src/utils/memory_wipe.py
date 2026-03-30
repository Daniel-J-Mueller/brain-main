"""Utility for clearing all persistent state."""

from pathlib import Path
import shutil

try:  # Support running as a script
    from .logger import get_logger
    from .config import load_config, BASE_DIR
except ImportError:  # pragma: no cover - fallback when executed directly
    from logger import get_logger
    from config import load_config, BASE_DIR


def wipe(persist_dir: str | Path | None = None) -> None:
    """Delete saved memory snapshots in the persistent directory."""
    cfg = load_config("configs/default.yaml")
    settings = cfg.get("settings", {})
    recalc_tables = bool(settings.get("recalculate_lookup_tables", False))

    if persist_dir:
        path = Path(persist_dir)
    else:
        path = Path(cfg.get("persistent_dir", "persistent"))
        if not path.is_absolute():
            path = BASE_DIR / path

    logger = get_logger("memory_wipe")

    if not path.exists():
        logger.info(f"{path} does not exist; nothing to wipe")
        return

    removed = False
    for filename in (
        "hippocampus_memory.npy",
        "hippocampus_memory.npz",
        "motor_cortex_adapters.pt",
        "wernicke_adapter.pt",
        "insular_mapping.pt",
        "motor_insula.pt",
        "frontal_lobe.pt",
        "amygdala_emotion.pt",
    ):
        target = path / filename
        if target.exists():
            target.unlink()
            logger.info(f"deleted {target}")
            removed = True

    # remove distributed hippocampus shards
    for shard in path.glob("hippocampus_memory_shard_*.npz"):
        shard.unlink()
        logger.info(f"deleted {shard}")
        removed = True

    # remove any additional files or directories except README
    for extra in list(path.iterdir()):
        if extra.name == "README.md":
            continue
        if not recalc_tables and extra.name in ("token_embeddings.npy", "valence.npy"):
            continue
        if extra.is_file():
            extra.unlink()
            logger.info(f"removed file {extra}")
            removed = True
        elif extra.is_dir():
            shutil.rmtree(extra)
            logger.info(f"removed directory {extra}")
            removed = True

    if removed:
        logger.info("memory wiped")
    else:
        logger.info("no memory snapshots found")

    log_path = Path(cfg.get("log_dir", "logs"))
    if not log_path.is_absolute():
        log_path = BASE_DIR / log_path
    if log_path.exists():
        log_removed = False
        for item in log_path.iterdir():
            if item.name == ".gitkeep":
                continue
            if item.is_file():
                item.unlink()
                logger.info(f"removed log file {item}")
                log_removed = True
            elif item.is_dir():
                shutil.rmtree(item)
                logger.info(f"removed log directory {item}")
                log_removed = True
        if log_removed:
            logger.info("logs cleared")
        else:
            logger.info("no log files found")
    else:
        logger.info(f"{log_path} does not exist; nothing to wipe")


if __name__ == "__main__":
    wipe()
