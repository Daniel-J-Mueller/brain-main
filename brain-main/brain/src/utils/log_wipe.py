"""Utility to clear log files from the log directory."""

from pathlib import Path
import shutil

try:
    from .logger import get_logger
except ImportError:  # pragma: no cover
    from logger import get_logger


def wipe(log_dir: str | Path | None = None) -> None:
    path = Path(log_dir) if log_dir else Path(__file__).resolve().parents[2] / "logs"
    logger = get_logger("log_wipe")

    if not path.exists():
        logger.info(f"{path} does not exist; nothing to wipe")
        return

    removed = False
    for item in path.iterdir():
        if item.is_file():
            item.unlink()
            logger.info(f"removed {item}")
            removed = True
        elif item.is_dir():
            shutil.rmtree(item)
            logger.info(f"removed directory {item}")
            removed = True
    if removed:
        logger.info("logs cleared")
    else:
        logger.info("no log files found")


if __name__ == "__main__":
    wipe()
