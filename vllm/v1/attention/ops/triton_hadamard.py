import math

import torch


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 3, "expected (tokens, heads, head_size)"
    head_size = x.shape[-1]
    assert head_size & (head_size - 1) == 0, "head_size must be power of 2"
    assert head_size in (64, 128), (
        f"only support 64 or 128 head size, but got {head_size}"
    )

    y = x.float().clone()
    h = 1
    while h < head_size:
        v = y.view(*y.shape[:-1], -1, 2, h)
        a = v[..., 0, :].clone()
        b = v[..., 1, :].clone()
        v[..., 0, :] = a + b
        v[..., 1, :] = a - b
        h *= 2

    return (y / math.sqrt(head_size)).to(x.dtype)
