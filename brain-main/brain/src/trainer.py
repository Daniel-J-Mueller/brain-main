"""Online adaptation utilities for brain."""

from typing import Iterable

import torch
from torch import nn

from .utils.sentinel import SENTINEL, UNTRAINED_INIT


class Trainer:
    """Very small placeholder trainer implementing simple Hebbian-like updates."""

    def __init__(self, lr: float = 1e-4, decay: float = 0.999, novelty_alpha: float = 0.9):
        self.lr = lr
        self.decay = decay
        self.novelty_alpha = novelty_alpha
        self._prev_activation: torch.Tensor | None = None

    def reset(self) -> None:
        """Clear any stored activation history."""
        self._prev_activation = None

    @torch.no_grad()
    def step(
        self,
        modules: Iterable[nn.Module],
        activations: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> None:
        """Apply a trivial Hebbian update to all adapter weights.

        Parameters
        ----------
        modules:
            Collection of modules containing parameters to update.
        activations:
            Activation tensor used to compute outer-product updates.
        lr_scale:
            Multiplier applied to ``self.lr`` for this step. Allows temporary
            learning-rate boosts when receiving corrective feedback.
        """
        outer = torch.einsum("bi,bj->ij", activations, activations)
        act = activations.squeeze(0)

        # Compute novelty based on similarity to recent activation
        novelty = 1.0
        if self._prev_activation is not None:
            prev = self._prev_activation.to(act.device)
            sim = torch.nn.functional.cosine_similarity(act, prev, dim=0)
            novelty = float(1.0 - sim.clamp(min=0.0).item())

            # Update running average of activation
            self._prev_activation.mul_(self.novelty_alpha)
            self._prev_activation.add_((1.0 - self.novelty_alpha) * act.cpu())
        else:
            self._prev_activation = act.cpu().clone()

        scaled_lr = self.lr * lr_scale * novelty
        for module in modules:
            for p in module.parameters():
                if not p.requires_grad:
                    continue

                data = p.data
                sentinel_mask = data == SENTINEL
                data[~sentinel_mask] *= self.decay

                if p.ndim == 0:
                    update = scaled_lr * act.to(data.device).mean()
                    if sentinel_mask.item():
                        data.copy_(UNTRAINED_INIT * update)
                    else:
                        data.add_(update)
                elif p.ndim == 1:
                    length = min(data.shape[0], act.shape[0])
                    update = scaled_lr * act.to(data.device)[:length]
                    mask = sentinel_mask[:length]
                    if mask.any():
                        data[:length][mask] = UNTRAINED_INIT * update[mask]
                    data[:length][~mask] += update[~mask]
                else:
                    rows = min(data.shape[0], outer.shape[0])
                    cols = min(data.shape[1], outer.shape[1])
                    update = scaled_lr * outer.to(data.device)[:rows, :cols]
                    mask = sentinel_mask[:rows, :cols]
                    if mask.any():
                        data[:rows, :cols][mask] = UNTRAINED_INIT * update[mask]
                    data[:rows, :cols][~mask] += update[~mask]

    @torch.no_grad()
    def align(
        self,
        modules: Iterable[nn.Module],
        target: torch.Tensor,
        actual: torch.Tensor,
        lr_scale: float = 1.0,
    ) -> None:
        """Adjust parameters to make ``actual`` closer to ``target``.

        ``target`` and ``actual`` may originate from different devices
        (e.g. DMN vs. motor cortex).  ``actual`` is therefore moved to the
        device of ``target`` before computing the error so that operations
        succeed regardless of their source locations.

        lr_scale:
            Multiplier applied to ``self.lr`` for this alignment update.
        """

        if target.device != actual.device:
            actual = actual.to(target.device)

        t = target
        a = actual
        if t.dim() > 2:
            t = t.mean(dim=1)
        if a.dim() > 2:
            a = a.mean(dim=1)

        error = t - a
        adjust = torch.einsum("bi,bj->ij", t, error)
        for module in modules:
            for p in module.parameters():
                if not p.requires_grad:
                    continue

                data = p.data
                sentinel_mask = data == SENTINEL
                data[~sentinel_mask] *= self.decay

                if p.ndim == 0:
                    grad = error.mean().to(data.device)
                    upd = self.lr * lr_scale * grad
                    if sentinel_mask.item():
                        data.copy_(UNTRAINED_INIT * upd)
                    else:
                        data.add_(upd)
                elif p.ndim == 1:
                    length = min(data.shape[0], error.shape[1])
                    grad = error.mean(dim=0).to(data.device)[:length]
                    mask = sentinel_mask[:length]
                    if mask.any():
                        data[:length][mask] = UNTRAINED_INIT * grad[mask]
                    data[:length][~mask] += self.lr * lr_scale * grad[~mask]
                else:
                    rows = min(data.shape[0], adjust.shape[0])
                    cols = min(data.shape[1], adjust.shape[1])
                    upd = self.lr * lr_scale * adjust.to(data.device)[:rows, :cols]
                    mask = sentinel_mask[:rows, :cols]
                    if mask.any():
                        data[:rows, :cols][mask] = UNTRAINED_INIT * upd[mask]
                    data[:rows, :cols][~mask] += upd[~mask]


if __name__ == "__main__":
    # Minimal smoke test
    trainer = Trainer()
    lin = nn.Linear(4, 4)
    act = torch.randn(1, 4)
    trainer.step([lin], act)
    print("updated", lin.weight.norm().item())
