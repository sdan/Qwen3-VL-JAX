import modal
from pathlib import Path

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install("triton>=3.0.0")
    .add_local_dir(local_dir, remote_path="/root/kernels")
)

@app.function(image=base_image, gpu="H100", timeout=600)
def run_v13():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3-VL vision encoder config
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    depth = 27  # number of vision transformer layers
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    print("=" * 70)
    print("RoPE Caching: Correct Pattern vs Naive Pattern")
    print("=" * 70)

    from vision_attention import TritonVisionAttention, compute_vision_rope_2d
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    for img_size in [448, 896, 1344]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        print(f"\n{'='*70}")
        print(f"Image: {img_size}x{img_size} -> seq_len={seq_len}, depth={depth} layers")
        print("=" * 70)

        # Get window indices
        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

        # Create test tensors
        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

        # Initialize module (shared across all layers in real model)
        attn = TritonVisionAttention(
            hidden_size, num_heads, head_dim,
            spatial_merge_size=spatial_merge_size
        ).to(device, dtype)

        # ============================================================
        # NAIVE PATTERN: Recompute RoPE each layer (what we were doing)
        # ============================================================
        def forward_naive():
            """Simulates 27 layers, each recomputing RoPE"""
            xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
            out = xs
            for _ in range(depth):
                # Each layer recomputes RoPE internally (cos/sin not passed)
                out = attn(out, cu_seqlens=cu_seqlens, grid_thw=grid_thw, window_idx=window_index)
            return window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)

        # ============================================================
        # CORRECT PATTERN: Compute RoPE once, pass to all layers
        # ============================================================
        def forward_cached():
            """Simulates 27 layers with RoPE computed once"""
            xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)

            # Compute RoPE ONCE (like JAX model does)
            cos, sin = compute_vision_rope_2d(
                grid_thw,
                head_dim=head_dim,
                spatial_merge_size=spatial_merge_size,
                rope_theta=10000.0,
                device=device,
                dtype=dtype,
                window_idx=window_index,
            )

            out = xs
            for _ in range(depth):
                # Pass precomputed cos/sin - no recomputation!
                out = attn(out, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            return window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)

        # Warmup
        for _ in range(3):
            forward_naive()
            forward_cached()
        torch.cuda.synchronize()

        # Benchmark
        naive_ms = triton.testing.do_bench(forward_naive, warmup=10, rep=50)
        cached_ms = triton.testing.do_bench(forward_cached, warmup=10, rep=50)

        speedup = naive_ms / cached_ms
        savings = naive_ms - cached_ms

        print(f"\nNaive (recompute RoPE each layer): {naive_ms:.3f} ms")
        print(f"Cached (compute RoPE once):         {cached_ms:.3f} ms")
        print(f"Speedup:                            {speedup:.2f}x")
        print(f"Time saved:                         {savings:.3f} ms ({100*savings/naive_ms:.1f}%)")

        # Per-layer breakdown
        per_layer_naive = naive_ms / depth
        per_layer_cached = cached_ms / depth
        print(f"\nPer-layer: {per_layer_naive:.3f} ms (naive) vs {per_layer_cached:.3f} ms (cached)")

        # RoPE compute time (for reference)
        def rope_only():
            return compute_vision_rope_2d(
                grid_thw,
                head_dim=head_dim,
                spatial_merge_size=spatial_merge_size,
                rope_theta=10000.0,
                device=device,
                dtype=dtype,
                window_idx=window_index,
            )

        rope_ms = triton.testing.do_bench(rope_only, warmup=20, rep=100)
        print(f"RoPE compute time (once):           {rope_ms:.3f} ms")
        print(f"RoPE compute time (27x naive):      {rope_ms * depth:.3f} ms")

    print("\n" + "=" * 70)
    print("Conclusion: Pass precomputed cos/sin to forward() for ~30% speedup")
    print("=" * 70)

    return True


if __name__ == "__main__":
    with app.run():
        run_v13.remote()
