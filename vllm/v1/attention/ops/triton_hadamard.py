import torch

from vllm.triton_utils import tl, triton

# This is for (64, 128) head sizes, for larger, will need CUDA kernels


@triton.jit
def hadamard_kernel(x_ptr, out_ptr, stride_token: tl.int64,
                    stride_head: tl.int64, HEAD_SIZE: tl.constexpr):
    tl.static_assert(
        (HEAD_SIZE & (HEAD_SIZE - 1)) == 0,
        "HEAD_SIZE must be power of 2",
    )

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    offs = tl.arange(0, HEAD_SIZE)
    base = token_idx * stride_token + head_idx * stride_head
    x = tl.load(x_ptr + base + offs).to(tl.float32)
    tl.store(out_ptr + base + offs, x)

    for i in tl.static_range(0, 7):
        span = 1 << i
        if span < HEAD_SIZE:
            even = (offs // span) % 2 == 0
            partner = tl.where(even, offs + span, offs - span)
            x = tl.load(out_ptr + base + offs)
            x_partner = tl.load(out_ptr + base + partner)
            result = tl.where(even, x + x_partner, x_partner - x)
            tl.store(out_ptr + base + offs, result)

    x = tl.load(out_ptr + base + offs)
    x = x / tl.sqrt(tl.cast(HEAD_SIZE, tl.float32))
    tl.store(out_ptr + base + offs, x)


def hadamard_rotate(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 3, "tokens, head, size"
    num_tokens, num_heads, head_size = x.shape
    assert head_size in (64, 128), (
        f"only support 64 or 128 head size, but got {head_size}"
    )
    assert head_size & (head_size - 1) == 0

    out = torch.empty_like(x)
    grid = (num_tokens, num_heads)
    hadamard_kernel[grid](
        x,
        out,
        x.stride(0),
        x.stride(1),
        HEAD_SIZE=head_size,
        num_warps=4 if head_size == 64 else 8,
        num_stages=2,
    )

    return out
