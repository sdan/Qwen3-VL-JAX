from __future__ import annotations

import torch

try:
    from flash_attn import flash_attn_varlen_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


def get_window_index(
    grid_thw: torch.Tensor | list[tuple[int, int, int]],
    *,
    window_size: int = 112,
    spatial_merge_size: int = 2,
    patch_size: int = 14,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Qwen3-VL vision window shuffle indices.

    This matches `model.py::Qwen3VisionTransformer._get_window_index`:
      - window_index is a permutation over merge positions (length = seq_len / merge_unit)
      - cu_seqlens is in token space (lengths * merge_unit), suitable for varlen window attention
    """
    if isinstance(grid_thw, torch.Tensor):
        if grid_thw.ndim != 2 or grid_thw.shape[-1] != 3:
            raise ValueError(f"grid_thw must be [N,3], got {tuple(grid_thw.shape)}")
        thw_list = [(int(t), int(h), int(w)) for t, h, w in grid_thw.tolist()]
        if device is None:
            device = grid_thw.device
    else:
        thw_list = [(int(t), int(h), int(w)) for (t, h, w) in grid_thw]
        if device is None:
            device = torch.device("cpu")

    if not thw_list:
        raise ValueError("grid_thw must be non-empty")
    if window_size <= 0:
        raise ValueError(f"window_size must be > 0, got {window_size}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be > 0, got {spatial_merge_size}")
    if patch_size <= 0:
        raise ValueError(f"patch_size must be > 0, got {patch_size}")

    vit_window = window_size // spatial_merge_size // patch_size
    if vit_window <= 0:
        raise ValueError(
            "vit_window computed as window_size//spatial_merge_size//patch_size must be >= 1, "
            f"got {vit_window} (window_size={window_size} spatial_merge_size={spatial_merge_size} patch_size={patch_size})"
        )

    merge_unit = spatial_merge_size**2
    window_indices: list[torch.Tensor] = []
    cu_seqlens: list[int] = [0]
    window_id = 0

    for t, h, w in thw_list:
        if t <= 0:
            raise ValueError(f"grid_thw temporal size must be > 0, got t={t}")
        if h <= 0 or w <= 0:
            raise ValueError(f"grid_thw h/w must be > 0, got h={h} w={w}")
        if h % spatial_merge_size != 0 or w % spatial_merge_size != 0:
            raise ValueError(
                f"h and w must be divisible by spatial_merge_size={spatial_merge_size}, got h={h} w={w}"
            )

        gh = h // spatial_merge_size
        gw = w // spatial_merge_size

        index = torch.arange(t * gh * gw, device=device, dtype=torch.int64).reshape(
            t, gh, gw
        )

        pad_h = (vit_window - (gh % vit_window)) % vit_window
        pad_w = (vit_window - (gw % vit_window)) % vit_window
        num_h = (gh + pad_h) // vit_window
        num_w = (gw + pad_w) // vit_window

        index_pad = torch.nn.functional.pad(index, (0, pad_w, 0, pad_h), value=-100)
        index_pad = index_pad.reshape(t, num_h, vit_window, num_w, vit_window)
        index_pad = index_pad.permute(0, 1, 3, 2, 4).reshape(
            t, num_h * num_w, vit_window, vit_window
        )

        seqlens = (index_pad != -100).sum(dim=(2, 3)).reshape(-1)
        index_new = index_pad.reshape(-1)
        index_new = index_new[index_new != -100]

        window_indices.append(index_new + window_id)

        cu_tmp = torch.cumsum(seqlens, dim=0) * merge_unit + cu_seqlens[-1]
        cu_seqlens.extend(int(x) for x in cu_tmp.tolist())

        window_id += int(t * gh * gw)

    window_index = torch.cat(window_indices, dim=0).to(device=device, dtype=torch.long)
    cu_arr = torch.tensor(cu_seqlens, dtype=torch.int32, device=device)

    # Remove duplicates (mirrors JAX `mask = [True] + diff != 0`).
    mask = torch.cat([torch.tensor([True], device=device), cu_arr[1:] != cu_arr[:-1]])
    return window_index.contiguous(), cu_arr[mask].contiguous()


def window_shuffle(
    x: torch.Tensor,
    window_index: torch.Tensor,
    *,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Shuffle tokens into window order (mirrors `model.py` window shuffle)."""
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim, got {tuple(x.shape)}")
    merge_unit = int(spatial_merge_size) ** 2
    if merge_unit <= 0:
        raise ValueError(f"spatial_merge_size must be > 0, got {spatial_merge_size}")

    seq_len = int(x.shape[0])
    if seq_len % merge_unit != 0:
        raise ValueError(f"seq_len must be divisible by merge_unit={merge_unit}, got {seq_len}")
    n_merge = seq_len // merge_unit
    if window_index.numel() != n_merge:
        raise ValueError(f"window_index length must be {n_merge}, got {window_index.numel()}")

    x_view = x.reshape(n_merge, merge_unit, *x.shape[1:])
    return x_view[window_index].reshape(seq_len, *x.shape[1:])


def window_unshuffle(
    x: torch.Tensor,
    window_index: torch.Tensor,
    *,
    spatial_merge_size: int = 2,
) -> torch.Tensor:
    """Inverse of `window_shuffle` (unshuffle back to original token order)."""
    if x.ndim < 1:
        raise ValueError(f"x must have at least 1 dim, got {tuple(x.shape)}")
    merge_unit = int(spatial_merge_size) ** 2
    if merge_unit <= 0:
        raise ValueError(f"spatial_merge_size must be > 0, got {spatial_merge_size}")

    seq_len = int(x.shape[0])
    if seq_len % merge_unit != 0:
        raise ValueError(f"seq_len must be divisible by merge_unit={merge_unit}, got {seq_len}")
    n_merge = seq_len // merge_unit
    if window_index.numel() != n_merge:
        raise ValueError(f"window_index length must be {n_merge}, got {window_index.numel()}")

    inv = torch.empty_like(window_index)
    inv[window_index] = torch.arange(n_merge, device=window_index.device, dtype=window_index.dtype)

    x_view = x.reshape(n_merge, merge_unit, *x.shape[1:])
    return x_view[inv].reshape(seq_len, *x.shape[1:])


def flash_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_heads: int,
    head_dim: int,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Windowed attention using flash_attn_varlen_func.

    All windows processed in parallel on GPU via packed varlen API.
    Expects q/k/v already shuffled into window order.
    """
    if not HAS_FLASH_ATTN:
        raise RuntimeError("flash_attn not available")

    seq_len = q.shape[0]
    q = q.view(seq_len, num_heads, head_dim)
    k = k.view(seq_len, num_heads, head_dim)
    v = v.view(seq_len, num_heads, head_dim)

    window_lens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seqlen = int(window_lens.max().item())

    if softmax_scale is None:
        softmax_scale = head_dim ** -0.5

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=False,
    )

    return out.view(seq_len, num_heads * head_dim)

