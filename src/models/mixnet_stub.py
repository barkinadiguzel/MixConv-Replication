import torch
import torch.nn as nn

from src.blocks.inverted_residual import InvertedResidualMixConv


class MixNetStub(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.stage1 = InvertedResidualMixConv(
            in_channels=16,
            out_channels=24,
            expansion=6,
            kernel_sizes=[3, 5],
            stride=2,
        )

        self.stage2 = InvertedResidualMixConv(
            in_channels=24,
            out_channels=40,
            expansion=6,
            kernel_sizes=[3, 5, 7],
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x
