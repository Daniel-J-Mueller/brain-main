import torch
from torch import nn
from pathlib import Path


class FatigueLoRA(nn.Module):
    """LoRA adapter with short-term fatigue dynamics."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        r: int = 4,
        decay: float = 0.9,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.register_buffer("fatigue", torch.ones(out_dim))
        self.register_buffer("recovery", torch.rand(out_dim) * 0.01 + 0.005)
        self.decay = decay
        self.r = r
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.A.t())
        out = torch.matmul(out, self.B.t()) / max(1, self.r)
        fatigue = self.fatigue.view(*([1] * (out.dim() - 1)), -1)
        out = out * fatigue
        with torch.no_grad():
            dims = tuple(range(out.dim() - 1))
            usage = out.abs().mean(dim=dims).view(-1)
            decay_factor = torch.pow(
                torch.as_tensor(self.decay, device=usage.device), usage
            )
            self.fatigue.mul_(decay_factor)
            self.fatigue.add_(self.recovery)
            self.fatigue.clamp_(0.0, 1.0)
        return out


class LongTermLoRA(nn.Module):
    """Simple LoRA adapter for long-term filtering."""

    def __init__(
        self, in_dim: int, out_dim: int, r: int = 4, device: str = "cpu"
    ) -> None:
        super().__init__()
        self.A = nn.Parameter(torch.zeros(r, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, r))
        self.r = r
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(x, self.A.t())
        out = torch.matmul(out, self.B.t()) / max(1, self.r)
        return out


def save_loras(module: nn.Module, base_path: str | Path) -> None:
    """Persist any direct LoRA children of ``module``."""

    base = Path(base_path)
    stem = base.stem
    for name, child in module.named_children():
        if isinstance(child, (FatigueLoRA, LongTermLoRA)):
            path = base.with_name(f"{stem}_{name}.pt")
            torch.save(child.state_dict(), path)

