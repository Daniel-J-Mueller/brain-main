import torch
from pathlib import Path


def model_memory_mb(model: torch.nn.Module) -> float:
    """Return the GPU memory usage of ``model`` parameters and buffers."""
    total = 0
    params = getattr(model, "parameters", None)
    if callable(params):
        for p in params():
            if p.device.type == "cuda":
                total += p.numel() * p.element_size()
    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        for b in buffers():
            if b.device.type == "cuda":
                total += b.numel() * b.element_size()
    return total / (1024 * 1024)


def log_model_memory(model: torch.nn.Module, name: str, log_dir: Path) -> None:
    """Append GPU memory usage for ``model`` to ``log_dir/GPU_debug.log``."""
    params = getattr(model, "parameters", None)
    buffers = getattr(model, "buffers", None)
    if not callable(params) and not callable(buffers):
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    mb = model_memory_mb(model)
    log_path = log_dir / "GPU_debug.log"
    with log_path.open("a") as f:
        f.write(f"{name}: {mb:.2f} MB\n")


def log_device_memory(device: str, log_dir: Path) -> None:
    """Append current allocated/reserved memory for ``device``."""
    if not torch.cuda.is_available():
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
    log_path = log_dir / "GPU_debug.log"
    with log_path.open("a") as f:
        f.write(f"Device {device}: allocated={alloc:.2f} MB reserved={reserved:.2f} MB\n")
