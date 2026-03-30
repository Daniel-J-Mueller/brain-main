import yaml
from pathlib import Path
from typing import Any, Dict

# Base directory of the ``brain`` package. Configuration files are stored
# relative to this path.
BASE_DIR = Path(__file__).resolve().parents[2]


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path : str
        Path to the YAML file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = BASE_DIR / cfg_path

    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)

    def resolve(p: str) -> str:
        path = Path(p)
        if not path.is_absolute():
            path = BASE_DIR / path
        return str(path)

    if "models" in cfg:
        cfg["models"] = {k: resolve(v) for k, v in cfg["models"].items()}

    if "persistent_dir" in cfg:
        cfg["persistent_dir"] = resolve(cfg["persistent_dir"])

    if "log_dir" in cfg:
        cfg["log_dir"] = resolve(cfg["log_dir"])

    return cfg
