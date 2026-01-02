"""Triton kernel for Qwen3-VL Vision Attention

Fused window attention + 3D RoPE for vision encoder.
This is the core kernel that makes Qwen3-VL vision encoding fast.

Key optimizations:
1. Fused QKV projection + RoPE application
2. Tiled attention computation (FlashAttention-style)
3. Window-based attention patterns
4. Memory-efficient softmax
5. Support for non-power-of-2 head_dim (e.g., 72 for Qwen3-VL)

Author: Built for Modal H100 deployment
"""
import torch
import triton
import triton.language as tl
import math
from collections import OrderedDict


def next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


_VISION_ROPE_CACHE_MAX_ITEMS = 32
_vision_rope_cache = OrderedDict()


def clear_vision_rope_cache() -> None:
    """Clear cached 2D vision RoPE tables."""
    _vision_rope_cache.clear()


def _canonical_device_key(device: torch.device) -> tuple[str, int | None]:
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
    else:
        idx = device.index
    return device.type, idx


def compute_vision_rope_2d(
    grid_thw,
    *,
    head_dim: int,
    spatial_merge_size: int,
    rope_theta: float,
    device,
    dtype,
    window_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL-style 2D RoPE (h, w) tables for vision tokens.

    Returns (cos, sin) with shape [total_seq, head_dim//2] suitable for the
    fused Triton RoPE+attention kernels (pairwise rotation between halves).

    grid_thw may be a torch tensor shaped [N, 3] or an iterable of (t, h, w).
    h and w are patch-grid sizes before the final patch merger; positions are
    ordered to match the spatial_merge_size grouping used by the model.

    If window_idx is provided, it should index merge positions (length
    total_seq // (spatial_merge_size**2)), and the tables are reordered to
    match the model's window shuffling.
    """
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"head_dim must be positive and even, got {head_dim}")
    rotary_dim = head_dim // 2
    if rotary_dim % 2 != 0:
        raise ValueError(
            f"head_dim//2 must be even for 2D RoPE, got head_dim={head_dim}"
        )
    merge = int(spatial_merge_size)
    if merge <= 0:
        raise ValueError(f"spatial_merge_size must be > 0, got {spatial_merge_size}")

    device = torch.device(device)

    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"grid_thw must be [N,3], got {tuple(grid_thw.shape)}")
        if grid_thw.is_cuda:
            grid_thw = grid_thw.detach().cpu()
        thw_list = [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
    else:
        thw_list = [(int(t), int(h), int(w)) for (t, h, w) in grid_thw]
    if not thw_list:
        raise ValueError("grid_thw must be non-empty")

    dev_type, dev_index = _canonical_device_key(device)
    cache_key = (
        tuple(thw_list),
        int(head_dim),
        int(merge),
        float(rope_theta),
        dev_type,
        dev_index,
        dtype,
    )

    cached = _vision_rope_cache.get(cache_key)
    if cached is not None:
        _vision_rope_cache.move_to_end(cache_key)
        cos_base, sin_base = cached
    else:
        inv_freq = 1.0 / (
            float(rope_theta)
            ** (
                torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
                / float(rotary_dim)
            )
        )  # [rotary_dim//2]

        hpos_all, wpos_all = [], []
        for t, h, w in thw_list:
            if t <= 0:
                raise ValueError(f"grid_thw temporal size must be > 0, got t={t}")
            if h <= 0 or w <= 0:
                raise ValueError(f"grid_thw h/w must be > 0, got h={h} w={w}")
            if h % merge != 0 or w % merge != 0:
                raise ValueError(
                    f"h and w must be divisible by spatial_merge_size={merge}, got h={h} w={w}"
                )

            hpos = torch.arange(h, device=device, dtype=torch.int32)[:, None].expand(h, w)
            wpos = torch.arange(w, device=device, dtype=torch.int32)[None, :].expand(h, w)

            hpos = (
                hpos.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )
            wpos = (
                wpos.reshape(h // merge, merge, w // merge, merge)
                .permute(0, 2, 1, 3)
                .reshape(-1)
            )

            if t > 1:
                hpos = hpos.repeat(t)
                wpos = wpos.repeat(t)

            hpos_all.append(hpos)
            wpos_all.append(wpos)

        hpos_all_f32 = torch.cat(hpos_all, dim=0).to(torch.float32)
        wpos_all_f32 = torch.cat(wpos_all, dim=0).to(torch.float32)

        freqs_h = hpos_all_f32[:, None] * inv_freq[None, :]
        freqs_w = wpos_all_f32[:, None] * inv_freq[None, :]
        rotary_emb = torch.cat([freqs_h, freqs_w], dim=-1)  # [seq, head_dim//2]
        cos_base = rotary_emb.cos().to(dtype).contiguous()
        sin_base = rotary_emb.sin().to(dtype).contiguous()

        _vision_rope_cache[cache_key] = (cos_base, sin_base)
        _vision_rope_cache.move_to_end(cache_key)
        while len(_vision_rope_cache) > _VISION_ROPE_CACHE_MAX_ITEMS:
            _vision_rope_cache.popitem(last=False)

    cos = cos_base
    sin = sin_base

    if window_idx is not None:
        merge_unit = merge * merge
        if cos.shape[0] % merge_unit != 0:
            raise ValueError(
                f"total_seq must be divisible by merge_unit={merge_unit}, got {cos.shape[0]}"
            )
        num_merge_pos = cos.shape[0] // merge_unit
        if window_idx.numel() != num_merge_pos:
            raise ValueError(
                f"window_idx length must be total_seq//merge_unit={num_merge_pos}, got {window_idx.numel()}"
            )
        wi = window_idx.to(device=device, dtype=torch.long)
        cos = cos.reshape(num_merge_pos, merge_unit, rotary_dim)[wi].reshape(-1, rotary_dim)
        sin = sin.reshape(num_merge_pos, merge_unit, rotary_dim)[wi].reshape(-1, rotary_dim)

    return cos.contiguous(), sin.contiguous()


@triton.jit
def _rotary_kernel(
    x_ptr, cos_ptr, sin_ptr, out_ptr,
    seq_len, head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Apply rotary embeddings in-place"""
    pid = tl.program_id(0)

    # Each program handles one position
    pos = pid
    if pos >= seq_len:
        return

    half_dim = head_dim // 2

    for i in range(half_dim):
        # Load x values
        idx1 = pos * head_dim + i
        idx2 = pos * head_dim + i + half_dim

        x1 = tl.load(x_ptr + idx1)
        x2 = tl.load(x_ptr + idx2)

        # Load cos/sin
        cos_val = tl.load(cos_ptr + pos * half_dim + i)
        sin_val = tl.load(sin_ptr + pos * half_dim + i)

        # Apply rotation
        out1 = x1 * cos_val - x2 * sin_val
        out2 = x1 * sin_val + x2 * cos_val

        tl.store(out_ptr + idx1, out1)
        tl.store(out_ptr + idx2, out2)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ],
    key=["N_CTX", "PADDED_HEAD_DIM"],
)
@triton.jit
def _fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,  # Next power of 2 >= HEAD_DIM
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Fused attention kernel (FlashAttention-style tiling)
    Supports non-power-of-2 head_dim via padding.

    Q, K, V: [batch, heads, seq, head_dim]
    Out: [batch, heads, seq, head_dim]
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Initialize offsets - use PADDED_HEAD_DIM for arange (must be power of 2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, PADDED_HEAD_DIM)
    mask_d = offs_d < HEAD_DIM  # Mask for actual head_dim

    # Pointers to Q, K, V for this batch and head
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk)
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)

    # Load Q block - stays in SRAM
    mask_m = offs_m < N_CTX
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize accumulator and max/sum for online softmax
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, PADDED_HEAD_DIM], dtype=tl.float32)

    # Scale factor in log2-space: exp2(x) matches exp(x / log2(e))
    LOG2E = 1.4426950408889634  # log2(e)
    qk_scale = LOG2E / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    # Loop over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)

        # Load K, V blocks
        mask_n = offs_n_curr < N_CTX
        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Compute QK^T
        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Apply causal mask (optional - vision doesn't need it usually)
        # qk = tl.where(offs_m[:, None] >= offs_n_curr[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        acc = tl.dot(p.to(v.dtype), v, acc)

        m_i = m_ij

        # Advance pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output (only HEAD_DIM elements, not padded)
    o_ptrs = Out + off_z * stride_oz + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ],
    key=["MAX_CTX", "PADDED_HEAD_DIM"],
)
@triton.jit
def _window_attention_kernel(
    Q, K, V, Out,
    cu_seqlens,  # Cumulative sequence lengths for windows
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_om, stride_oh, stride_ok,
    num_windows, num_heads,
    MAX_CTX,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,  # Next power of 2 >= HEAD_DIM
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Window-based attention (variable-length) for vision encoder.
    Supports non-power-of-2 head_dim via padding.

    Q, K, V: [total_seq, num_heads, head_dim] packed by windows
    cu_seqlens: [num_windows + 1] cumulative sequence lengths
    """
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)

    window_id = pid_nh // num_heads
    head_id = pid_nh % num_heads

    if window_id >= num_windows:
        return

    # Get window boundaries
    start = tl.load(cu_seqlens + window_id)
    end = tl.load(cu_seqlens + window_id + 1)
    window_size = end - start

    if window_size <= 0:
        return

    start_m = pid_m * BLOCK_M
    if start_m >= window_size:
        return

    # Process this window block - use PADDED_HEAD_DIM for arange
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, PADDED_HEAD_DIM)
    mask_d = offs_d < HEAD_DIM  # Mask for actual head_dim

    mask_m = offs_m < window_size

    # Load Q block - keep in SRAM
    q_ptrs = Q + (start + offs_m[:, None]) * stride_qm + head_id * stride_qh + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Initialize accumulators for online softmax
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, PADDED_HEAD_DIM], dtype=tl.float32)

    # Scale factor in log2-space (exp2)
    LOG2E = 1.4426950408889634  # log2(e)
    qk_scale = LOG2E / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    # Loop over K/V blocks inside the window
    for start_n in range(0, window_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < window_size

        k_ptrs = K + (start + offs_n_curr[:, None]) * stride_km + head_id * stride_kh + offs_d[None, :] * stride_kk
        v_ptrs = V + (start + offs_n_curr[:, None]) * stride_vm + head_id * stride_vh + offs_d[None, :] * stride_vk

        k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * qk_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij

    # Final normalization and store
    acc = acc / l_i[:, None]
    o_ptrs = Out + (start + offs_m[:, None]) * stride_om + head_id * stride_oh + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=1, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_stages=2, num_warps=8),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32}, num_stages=2, num_warps=4),
    ],
    key=["MAX_CTX", "PADDED_HEAD_DIM"],
)
@triton.jit
def _window_attention_rope_kernel(
    Q, K, V,
    COS, SIN,
    Out,
    cu_seqlens,
    stride_qm, stride_qh, stride_qk,
    stride_km, stride_kh, stride_kk,
    stride_vm, stride_vh, stride_vk,
    stride_cm, stride_ck,
    stride_om, stride_oh, stride_ok,
    num_windows, num_heads,
    MAX_CTX,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,  # Next power of 2 >= HEAD_DIM
    PADDED_HALF_DIM: tl.constexpr,  # Next power of 2 >= HEAD_DIM // 2
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Window-based attention with RoPE fused into Q/K loads.
    Supports non-power-of-2 head_dim via padding.

    Q, K, V: [total_seq, num_heads, head_dim] packed by windows
    COS, SIN: [total_seq, head_dim//2] rotary tables (packed the same way as Q/K)
    cu_seqlens: [num_windows + 1] cumulative sequence lengths
    """
    pid_m = tl.program_id(0)
    pid_nh = tl.program_id(1)

    window_id = pid_nh // num_heads
    head_id = pid_nh % num_heads

    if window_id >= num_windows:
        return

    start = tl.load(cu_seqlens + window_id)
    end = tl.load(cu_seqlens + window_id + 1)
    window_size = end - start
    if window_size <= 0:
        return

    start_m = pid_m * BLOCK_M
    if start_m >= window_size:
        return

    half = HEAD_DIM // 2
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, PADDED_HEAD_DIM)
    offs_dh = tl.arange(0, PADDED_HALF_DIM)
    mask_d = offs_d < HEAD_DIM  # Mask for actual head_dim
    mask_dh = offs_dh < half  # Mask for actual half head_dim

    mask_m = offs_m < window_size

    # Load Q halves
    q0_ptrs = Q + (start + offs_m[:, None]) * stride_qm + head_id * stride_qh + offs_dh[None, :] * stride_qk
    q1_ptrs = Q + (start + offs_m[:, None]) * stride_qm + head_id * stride_qh + (offs_dh[None, :] + half) * stride_qk
    q0 = tl.load(q0_ptrs, mask=mask_m[:, None] & mask_dh[None, :], other=0.0)
    q1 = tl.load(q1_ptrs, mask=mask_m[:, None] & mask_dh[None, :], other=0.0)

    # Load RoPE tables for Q positions
    cos_q_ptrs = COS + (start + offs_m[:, None]) * stride_cm + offs_dh[None, :] * stride_ck
    sin_q_ptrs = SIN + (start + offs_m[:, None]) * stride_cm + offs_dh[None, :] * stride_ck
    cos_q = tl.load(cos_q_ptrs, mask=mask_m[:, None] & mask_dh[None, :], other=0.0).to(q0.dtype)
    sin_q = tl.load(sin_q_ptrs, mask=mask_m[:, None] & mask_dh[None, :], other=0.0).to(q0.dtype)

    # Apply RoPE to Q
    q0_rope = (q0 * cos_q - q1 * sin_q).to(q0.dtype)
    q1_rope = (q0 * sin_q + q1 * cos_q).to(q0.dtype)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, PADDED_HEAD_DIM], dtype=tl.float32)

    LOG2E = 1.4426950408889634  # log2(e)
    qk_scale = LOG2E / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))

    for start_n in range(0, window_size, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        offs_n_curr = start_n + offs_n
        mask_n = offs_n_curr < window_size

        # Load K halves
        k0_ptrs = K + (start + offs_n_curr[:, None]) * stride_km + head_id * stride_kh + offs_dh[None, :] * stride_kk
        k1_ptrs = K + (start + offs_n_curr[:, None]) * stride_km + head_id * stride_kh + (offs_dh[None, :] + half) * stride_kk
        k0 = tl.load(k0_ptrs, mask=mask_n[:, None] & mask_dh[None, :], other=0.0)
        k1 = tl.load(k1_ptrs, mask=mask_n[:, None] & mask_dh[None, :], other=0.0)

        # Load RoPE tables for K positions
        cos_k_ptrs = COS + (start + offs_n_curr[:, None]) * stride_cm + offs_dh[None, :] * stride_ck
        sin_k_ptrs = SIN + (start + offs_n_curr[:, None]) * stride_cm + offs_dh[None, :] * stride_ck
        cos_k = tl.load(cos_k_ptrs, mask=mask_n[:, None] & mask_dh[None, :], other=0.0).to(k0.dtype)
        sin_k = tl.load(sin_k_ptrs, mask=mask_n[:, None] & mask_dh[None, :], other=0.0).to(k0.dtype)

        k0_rope = (k0 * cos_k - k1 * sin_k).to(k0.dtype)
        k1_rope = (k0 * sin_k + k1 * cos_k).to(k0.dtype)

        # QK^T (RoPE applied), scaled for exp2
        qk = (tl.dot(q0_rope, tl.trans(k0_rope)) + tl.dot(q1_rope, tl.trans(k1_rope))) * qk_scale
        qk = tl.where(mask_n[None, :], qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, axis=1)

        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Load V and accumulate
        v_ptrs = V + (start + offs_n_curr[:, None]) * stride_vm + head_id * stride_vh + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)

    acc = acc / l_i[:, None]
    o_ptrs = Out + (start + offs_m[:, None]) * stride_om + head_id * stride_oh + offs_d[None, :] * stride_ok
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_d[None, :])


class TritonVisionAttention(torch.nn.Module):
    """
    Triton-accelerated vision attention for Qwen3-VL

    Features:
    - Window-based attention
    - 2D RoPE (height, width) matching Qwen3-VL vision encoder
    - Fused softmax
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        spatial_merge_size: int = 2,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.spatial_merge_size = int(spatial_merge_size)
        self.rope_theta = float(rope_theta)

        # QKV projection
        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def compute_vision_2d_rope(
        self,
        grid_thw,
        device,
        dtype,
        window_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_vision_rope_2d(
            grid_thw,
            head_dim=self.head_dim,
            spatial_merge_size=self.spatial_merge_size,
            rope_theta=self.rope_theta,
            device=device,
            dtype=dtype,
            window_idx=window_idx,
        )

    def _compute_rotary_emb(self, seq_len: int, device, dtype):
        """Fallback: simple 1D rotary embeddings (for testing without grid info)"""
        dim = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        grid_thw: list[tuple[int, int, int]] | None = None,
        window_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [seq_len, hidden_size] - flattened vision tokens
            cu_seqlens: [num_windows + 1] - cumulative sequence lengths for windows
            cos, sin: [seq_len, head_dim // 2] - precomputed rotary tables
            grid_thw: List of (t, h, w) tuples for 2D RoPE computation
                      If provided and cos/sin are None, computes proper 2D vision RoPE

        Returns:
            [seq_len, hidden_size]
        """
        seq_len = x.shape[0]

        # QKV projection
        qkv = self.qkv(x)  # [seq, 3*hidden]
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # Each: [seq, heads, head_dim]

        # RoPE tables (cos/sin): [seq, head_dim//2]
        if cos is None or sin is None:
            if grid_thw is not None:
                cos, sin = self.compute_vision_2d_rope(
                    grid_thw, x.device, x.dtype, window_idx=window_idx
                )
            else:
                cos, sin = self._compute_rotary_emb(seq_len, x.device, x.dtype)
        if cos.shape[0] != seq_len or sin.shape[0] != seq_len:
            raise ValueError(
                f"cos/sin must have first dim == seq_len ({seq_len}), "
                f"got cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
            )

        # Output tensor
        out = torch.empty_like(q)

        if cu_seqlens is not None:
            # Window attention mode
            num_windows = cu_seqlens.shape[0] - 1
            max_ctx = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

            # Compute padded dimensions for non-power-of-2 head_dim support
            padded_head_dim = next_power_of_2(self.head_dim)
            padded_half_dim = next_power_of_2(self.head_dim // 2)

            grid = lambda meta: (triton.cdiv(max_ctx, meta["BLOCK_M"]), num_windows * self.num_heads)
            _window_attention_rope_kernel[grid](
                q, k, v,
                cos, sin,
                out,
                cu_seqlens,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                cos.stride(0), cos.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                num_windows, self.num_heads,
                max_ctx,
                HEAD_DIM=self.head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
                PADDED_HALF_DIM=padded_half_dim,
            )
        else:
            # Treat as a single window (full attention).
            cu_seqlens = torch.tensor([0, seq_len], device=x.device, dtype=torch.int32)
            padded_head_dim = next_power_of_2(self.head_dim)
            padded_half_dim = next_power_of_2(self.head_dim // 2)

            grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), self.num_heads)
            _window_attention_rope_kernel[grid](
                q, k, v,
                cos, sin,
                out,
                cu_seqlens,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                cos.stride(0), cos.stride(1),
                out.stride(0), out.stride(1), out.stride(2),
                1, self.num_heads,
                seq_len,
                HEAD_DIM=self.head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
                PADDED_HALF_DIM=padded_half_dim,
            )

        # Reshape and project
        out = out.reshape(seq_len, self.hidden_size)
        return self.proj(out)


class UnfusedVisionAttention(torch.nn.Module):
    """
    Baseline attention path for comparison:
    - QKV projection (cuBLAS)
    - RoPE materialization in PyTorch
    - Triton window attention (no RoPE fused)
    - Output projection (cuBLAS)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        spatial_merge_size: int = 2,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.spatial_merge_size = int(spatial_merge_size)
        self.rope_theta = float(rope_theta)

        self.qkv = torch.nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    def compute_vision_2d_rope(
        self,
        grid_thw,
        device,
        dtype,
        window_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return compute_vision_rope_2d(
            grid_thw,
            head_dim=self.head_dim,
            spatial_merge_size=self.spatial_merge_size,
            rope_theta=self.rope_theta,
            device=device,
            dtype=dtype,
            window_idx=window_idx,
        )

    def _compute_rotary_emb(self, seq_len: int, device, dtype):
        dim = self.head_dim // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device).float() / dim))
        pos = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor = None,
        cos: torch.Tensor | None = None,
        sin: torch.Tensor | None = None,
        grid_thw: list[tuple[int, int, int]] | None = None,
        window_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[0]

        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        if cos is None or sin is None:
            if grid_thw is not None:
                cos, sin = self.compute_vision_2d_rope(
                    grid_thw, x.device, x.dtype, window_idx=window_idx
                )
            else:
                cos, sin = self._compute_rotary_emb(seq_len, x.device, x.dtype)
        if cos.shape[0] != seq_len or sin.shape[0] != seq_len:
            raise ValueError(
                f"cos/sin must have first dim == seq_len ({seq_len}), "
                f"got cos={tuple(cos.shape)} sin={tuple(sin.shape)}"
            )

        half = self.head_dim // 2
        cos_ = cos[:, None, :].to(dtype=q.dtype)
        sin_ = sin[:, None, :].to(dtype=q.dtype)

        q0, q1 = q[..., :half], q[..., half:]
        k0, k1 = k[..., :half], k[..., half:]
        q_rope = torch.cat([q0 * cos_ - q1 * sin_, q0 * sin_ + q1 * cos_], dim=-1)
        k_rope = torch.cat([k0 * cos_ - k1 * sin_, k0 * sin_ + k1 * cos_], dim=-1)

        out = torch.empty_like(q_rope)

        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0, seq_len], device=x.device, dtype=torch.int32)

        num_windows = cu_seqlens.shape[0] - 1
        max_ctx = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        padded_head_dim = next_power_of_2(self.head_dim)

        grid = lambda meta: (triton.cdiv(max_ctx, meta["BLOCK_M"]), num_windows * self.num_heads)
        _window_attention_kernel[grid](
            q_rope, k_rope, v,
            out,
            cu_seqlens,
            q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
            k_rope.stride(0), k_rope.stride(1), k_rope.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_windows, self.num_heads,
            max_ctx,
            HEAD_DIM=self.head_dim,
            PADDED_HEAD_DIM=padded_head_dim,
        )

        out = out.reshape(seq_len, self.hidden_size)
        return self.proj(out)


def benchmark_attention(seq_len=4096, hidden_size=1024, num_heads=16, num_runs=100):
    """Benchmark Triton attention vs PyTorch baseline"""
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Create module
    attn = TritonVisionAttention(hidden_size, num_heads).to(device, dtype)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        _ = attn(x)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        _ = attn(x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.perf_counter()

    ms_per_run = (end - start) / num_runs * 1000
    print(f"Triton Attention: {ms_per_run:.3f} ms/run")
    print(f"Seq len: {seq_len}, Hidden: {hidden_size}, Heads: {num_heads}")

    return ms_per_run


if __name__ == "__main__":
    benchmark_attention()
