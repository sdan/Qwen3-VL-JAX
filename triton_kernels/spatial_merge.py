import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_C": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 128}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_C": 256}, num_stages=2, num_warps=8),
    ],
    key=["C"],
)
@triton.jit
def _spatial_merge_2x2_kernel(
    X,
    Out,
    H,
    W,
    stride_xm,
    stride_xk,
    stride_om,
    stride_ok,
    C: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)

    out_w = W // 2
    out_h = pid // out_w
    out_w = pid - out_h * out_w

    in_h = out_h * 2
    in_w = out_w * 2

    in00 = in_h * W + in_w
    in01 = in00 + 1
    in10 = in00 + W
    in11 = in10 + 1

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x00 = tl.load(X + in00 * stride_xm + offs_c * stride_xk, mask=mask_c, other=0.0)
    x01 = tl.load(X + in01 * stride_xm + offs_c * stride_xk, mask=mask_c, other=0.0)
    x10 = tl.load(X + in10 * stride_xm + offs_c * stride_xk, mask=mask_c, other=0.0)
    x11 = tl.load(X + in11 * stride_xm + offs_c * stride_xk, mask=mask_c, other=0.0)

    out_base = Out + pid * stride_om
    tl.store(out_base + (offs_c + 0 * C) * stride_ok, x00, mask=mask_c)
    tl.store(out_base + (offs_c + 1 * C) * stride_ok, x01, mask=mask_c)
    tl.store(out_base + (offs_c + 2 * C) * stride_ok, x10, mask=mask_c)
    tl.store(out_base + (offs_c + 3 * C) * stride_ok, x11, mask=mask_c)


def spatial_merge_2x2(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("spatial_merge_2x2 expects a CUDA tensor")
    if x.ndim != 2:
        raise ValueError(f"spatial_merge_2x2 expects [H*W, C], got shape={tuple(x.shape)}")
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError("spatial_merge_2x2 requires even H and W")
    if x.shape[0] != h * w:
        raise ValueError(f"x.shape[0] must be H*W ({h*w}), got {x.shape[0]}")

    c = x.shape[1]
    out_tokens = (h // 2) * (w // 2)
    out = torch.empty((out_tokens, 4 * c), device=x.device, dtype=x.dtype)

    grid = lambda meta: (out_tokens, triton.cdiv(c, meta["BLOCK_C"]))
    _spatial_merge_2x2_kernel[grid](
        x,
        out,
        h,
        w,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        C=c,
    )
    return out

