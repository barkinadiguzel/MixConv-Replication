import torch
import torch.nn as nn

from src.blocks.mixconv_block import MixConvBlock


class MobileNetStub(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)

        self.block1 = MixConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_sizes=[3, 5],
            stride=1,
        )

        self.block2 = MixConvBlock(
            in_channels=64,
            out_channels=128,
            kernel_sizes=[3, 5, 7],
            stride=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        return x
