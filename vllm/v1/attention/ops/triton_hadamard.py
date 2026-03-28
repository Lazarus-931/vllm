import math

import torch

from vllm._custom_ops import hadacore_transform


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 3, "expected (tokens, heads, head_size)"
    head_size = x.shape[-1]
    assert head_size & (head_size - 1) == 0, "head_size must be power of 2"

    out = hadacore_transform(x, inplace=False)
    out.mul_(1.0 / math.sqrt(head_size))
    return out
