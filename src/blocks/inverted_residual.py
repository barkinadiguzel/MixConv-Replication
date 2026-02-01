import torch
import torch.nn as nn

from src.layers.mixconv import MixConv2d


class InvertedResidualMixConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        kernel_sizes: list[int],
        stride: int = 1,
    ):
        super().__init__()

        hidden_dim = in_channels * expansion
        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.mixconv = nn.Sequential(
            MixConv2d(
                in_channels=hidden_dim,
                kernel_sizes=kernel_sizes,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.mixconv(out)
        out = self.project(out)

        if self.use_residual:
            return x + out
        return out
