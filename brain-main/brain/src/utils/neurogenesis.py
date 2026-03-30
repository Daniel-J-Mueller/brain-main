from __future__ import annotations

"""Utilities for seeding neural regions when persistent weights are empty."""

from pathlib import Path
import json
from typing import Any

import torch
from torch import nn

__all__ = ["maybe_initialize"]


def _collect_tensors(obj: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(obj, dict):
        for v in obj.values():
            tensors.extend(_collect_tensors(v))
    elif isinstance(obj, torch.Tensor):
        tensors.append(obj)
    return tensors


def _is_blank_state(state: dict[str, Any]) -> bool:
    tensors = _collect_tensors(state)
    if not tensors:
        return True
    all_zero = all(float(t.abs().sum()) == 0.0 for t in tensors)
    all_const = all(float(t.var()) == 0.0 for t in tensors)
    return all_zero or all_const


def _record_birth(name: str, state_file: Path) -> None:
    state: dict[str, Any] = {}
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text())
        except Exception:
            state = {}
    state[name] = True
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state))


def _init_module(
    module: nn.Module, bias_shift: float = 0.0, var_scale: float = 1.0
) -> None:
    """Initialize parameters of ``module`` using Kaiming initialization."""

    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.mul_(var_scale)
                m.weight.add_(torch.randn_like(m.weight) * 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    if bias_shift:
                        m.bias.add_(bias_shift)
            if hasattr(m, "A") and hasattr(m, "B"):
                nn.init.normal_(m.A, std=0.02)
                nn.init.normal_(m.B, std=0.02)


def maybe_initialize(
    module: nn.Module,
    persist_path: str | Path | None,
    name: str,
    neurogenesis: bool,
    state_file: Path,
    *,
    bias_shift: float = 0.0,
    var_scale: float = 1.0,
) -> None:
    path = Path(persist_path) if persist_path else None
    born = False
    if path and path.exists():
        try:
            state = torch.load(path, map_location="cpu")
            if neurogenesis and _is_blank_state(state):
                _init_module(module, bias_shift=bias_shift, var_scale=var_scale)
                born = True
            else:
                module.load_state_dict(state, strict=False)
        except Exception:
            _init_module(module, bias_shift=bias_shift, var_scale=var_scale)
            born = True
    else:
        if neurogenesis:
            _init_module(module, bias_shift=bias_shift, var_scale=var_scale)
            born = True
    if born:
        _record_birth(name, state_file)
