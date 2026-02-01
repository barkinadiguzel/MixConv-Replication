import math
from typing import List, Dict


def split_channels(
    in_channels: int,
    kernel_sizes: List[int],
    mode: str = "equal"
) -> Dict[int, int]:
    num_kernels = len(kernel_sizes)

    if mode == "equal":
        base = in_channels // num_kernels
        remainder = in_channels % num_kernels

        splits = [base] * num_kernels
        for i in range(remainder):
            splits[i] += 1

    elif mode == "log":
        weights = [math.log(k) for k in kernel_sizes]
        total = sum(weights)
        splits = [int(in_channels * w / total) for w in weights]
        diff = in_channels - sum(splits)
        splits[0] += diff

    else:
        raise ValueError(f"Unknown split mode: {mode}")

    return {
        k: c for k, c in zip(kernel_sizes, splits)
    }
