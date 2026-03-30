import torch
from torch import nn

from .utils.sentinel import SentinelLinear

class OccipitalLobe(nn.Module):
    """Simple MLP over CLIP embeddings producing 128-dim visual features."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128, device: str = "cpu"):
        super().__init__()
        self.net = nn.Sequential(
            SentinelLinear(input_dim, hidden_dim),
            nn.ReLU(),
            SentinelLinear(hidden_dim, output_dim),
        )
        self.device = device
        self.to(device)

    @torch.no_grad()
    def process(self, embeddings: torch.Tensor) -> torch.Tensor:
        embeddings = embeddings.to(self.device)
        return self.net(embeddings)

if __name__ == "__main__":
    lobe = OccipitalLobe()
    dummy = torch.zeros(1, 512)
    print(lobe.process(dummy).shape)
