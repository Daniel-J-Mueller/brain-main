"""Multimodal fusion hub approximating the brain's DMN."""

from typing import Dict

import torch
from torch import nn

from .utils.sentinel import SentinelLinear


class DefaultModeNetwork(nn.Module):
    """Fuse sensory embeddings and produce routed context vectors."""

    def __init__(
        self,
        vision_dim: int = 128,
        audio_dim: int = 768,
        intero_dim: int = 64,
        hidden_dim: int = 2048,
        output_dim: int = 768,
        num_layers: int = 4,
        modality_weights: tuple[float, float, float] | None = None,
    ) -> None:
        """Create a larger fusion network.

        ``audio_dim`` defaults to 768 to match the hidden size of GPT-2 used in
        :class:`WernickesArea`. ``output_dim`` is likewise 768 so the resulting
        context can be fed directly to :class:`BrocasArea` without additional
        projection.
        """

        super().__init__()
        fusion_in = vision_dim + audio_dim + intero_dim
        self.fusion = SentinelLinear(fusion_in, hidden_dim)
        self.modality_scale = nn.Parameter(
            torch.tensor(modality_weights or (1.0, 1.0, 1.0), dtype=torch.float32)
        )

        layers = []
        for _ in range(num_layers - 1):
            layers.extend([SentinelLinear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(SentinelLinear(hidden_dim, output_dim))
        self.router = nn.Sequential(*layers)

    def set_modality_weights(self, vision: float, audio: float, intero: float) -> None:
        """Update modality weighting applied during fusion."""
        with torch.no_grad():
            self.modality_scale.copy_(torch.tensor([vision, audio, intero], device=self.modality_scale.device))

    @torch.no_grad()
    def forward(self, vision: torch.Tensor, audio: torch.Tensor, intero: torch.Tensor) -> torch.Tensor:
        """Return fused context embedding."""
        scale = torch.relu(self.modality_scale)
        weighted_vision = vision * scale[0]
        weighted_audio = audio * scale[1]
        weighted_intero = intero * scale[2]
        combined = torch.cat([weighted_vision, weighted_audio, weighted_intero], dim=-1)

        expected = self.fusion.in_features
        actual = combined.shape[-1]
        if actual != expected:
            if actual < expected:
                pad = expected - actual
                combined = nn.functional.pad(combined, (0, pad))
            else:
                combined = combined[..., :expected]

        # ``intero`` feedback may contain negative values intended to inhibit
        # repetition. ``ReLU`` would zero those out which defeats the purpose,
        # so use ``tanh`` here to preserve the sign while keeping the output
        # bounded.
        hidden = torch.tanh(self.fusion(combined))
        return self.router(hidden)
