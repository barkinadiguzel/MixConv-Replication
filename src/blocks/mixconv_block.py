import torch
import torch.nn as nn

from src.layers.mixconv import MixConv2d


class MixConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        stride: int = 1,
        activation: str = "relu",
    ):
        super().__init__()

        self.mixconv = MixConv2d(
            in_channels=in_channels,
            kernel_sizes=kernel_sizes,
            stride=stride,
            bias=False,
        )

        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )

        self.bn = nn.BatchNorm2d(out_channels)

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "swish":
            self.act = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixconv(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)
