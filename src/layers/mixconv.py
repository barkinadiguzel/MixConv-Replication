import torch
import torch.nn as nn
from typing import List

from src.kernels.kernel_sizes import split_channels


class MixConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_sizes: List[int],
        stride: int = 1,
        bias: bool = False,
        split_mode: str = "equal",
    ):
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.in_channels = in_channels

        channel_map = split_channels(
            in_channels=in_channels,
            kernel_sizes=kernel_sizes,
            mode=split_mode,
        )

        self.splits = list(channel_map.values())

        self.convs = nn.ModuleList()
        for k, c in channel_map.items():
            self.convs.append(
                nn.Conv2d(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=k,
                    stride=stride,
                    padding=k // 2,
                    groups=c,
                    bias=bias,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = torch.split(x, self.splits, dim=1)

        ys = []
        for x_i, conv in zip(xs, self.convs):
            ys.append(conv(x_i))
        return torch.cat(ys, dim=1)
