"""Synthetic video smoke test (no internet).

Qwen3-VL videos are treated as (T, H, W) patch grids. This test validates that
the window-indexing + RoPE + attention stack works for video-shaped grids.
"""

from __future__ import annotations

from pathlib import Path

import modal

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install("triton>=3.0.0")
    .add_local_dir(local_dir, remote_path="/root/kernels")
)


@app.function(image=base_image, gpu="H100", timeout=600)
def run_v17():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    from vision_attention import TritonVisionAttention, compute_vision_rope_2d
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    depth = 27

    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    num_frames = 16
    frame_h = 448
    frame_w = 448

    patches_h = frame_h // patch_size
    patches_w = frame_w // patch_size
    grid_thw = [(num_frames, patches_h, patches_w)]
    seq_len = int(num_frames * patches_h * patches_w)

    print("=" * 70)
    print("SYNTHETIC VIDEO: window attention smoke test")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}  dtype={dtype}")
    print(f"Frames: {num_frames}  Resolution: {frame_h}x{frame_w}")
    print(f"Grid (t,h,w): ({num_frames}, {patches_h}, {patches_w})  seq={seq_len}")

    window_index, cu_seqlens = get_window_index(
        grid_thw,
        window_size=window_size,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size,
        device=device,
    )
    assert cu_seqlens[0].item() == 0 and cu_seqlens[-1].item() == seq_len

    cos, sin = compute_vision_rope_2d(
        grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        rope_theta=10000.0,
        device=device,
        dtype=dtype,
        window_idx=window_index,
    )

    attn = TritonVisionAttention(
        hidden_size,
        num_heads,
        head_dim,
        spatial_merge_size=spatial_merge_size,
    ).to(device, dtype)

    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    def forward():
        xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
        out = xs
        for _ in range(depth):
            out = attn(out, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        out = window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)
        return out

    for _ in range(3):
        _ = forward()
    torch.cuda.synchronize()

    out = forward()
    torch.cuda.synchronize()
    assert out.shape == (seq_len, hidden_size)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()

    ms = triton.testing.do_bench(forward, warmup=10, rep=30)

    print("-" * 70)
    print(f"Total: {ms:.3f} ms  ({ms / num_frames:.3f} ms/frame)")
    print(f"Throughput: {num_frames / (ms / 1000):.2f} frames/sec")

    return {
        "frames": int(num_frames),
        "grid_thw": grid_thw[0],
        "seq_len": int(seq_len),
        "ms": float(ms),
        "ms_per_frame": float(ms / num_frames),
        "fps": float(num_frames / (ms / 1000)),
    }


if __name__ == "__main__":
    with app.run():
        run_v17.remote()
