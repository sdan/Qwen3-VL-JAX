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
def run_v14():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    depth = 27
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    print("=" * 70)
    print("Shuffle Overhead Analysis (Amortized Over 27 Layers)")
    print("=" * 70)

    from vision_attention import TritonVisionAttention, compute_vision_rope_2d
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    for img_size in [448, 896, 1344]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        seq_len = grid_thw[0][0] * grid_thw[0][1] * grid_thw[0][2]

        print(f"\n{'='*70}")
        print(f"Image: {img_size}x{img_size} -> seq_len={seq_len}")
        print("=" * 70)

        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

        attn = TritonVisionAttention(
            hidden_size, num_heads, head_dim,
            spatial_merge_size=spatial_merge_size
        ).to(device, dtype)

        # Precompute RoPE (correct pattern)
        cos, sin = compute_vision_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
            rope_theta=10000.0,
            device=device,
            dtype=dtype,
            window_idx=window_index,
        )

        results = {}

        # 1. Shuffle only
        def run_shuffle():
            return window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(10):
            run_shuffle()
        torch.cuda.synchronize()
        results["shuffle"] = triton.testing.do_bench(run_shuffle, warmup=20, rep=100)

        x_shuffled = run_shuffle()

        # 2. Unshuffle only
        def run_unshuffle():
            return window_unshuffle(x_shuffled, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(10):
            run_unshuffle()
        torch.cuda.synchronize()
        results["unshuffle"] = triton.testing.do_bench(run_unshuffle, warmup=20, rep=100)

        # 3. Single attention layer (with precomputed RoPE)
        def run_single_attn():
            return attn(x_shuffled, cu_seqlens=cu_seqlens, cos=cos, sin=sin)

        for _ in range(10):
            run_single_attn()
        torch.cuda.synchronize()
        results["single_attn"] = triton.testing.do_bench(run_single_attn, warmup=20, rep=100)

        # 4. Full 27-layer pass (correct pattern)
        def run_full_correct():
            xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
            out = xs
            for _ in range(depth):
                out = attn(out, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            return window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(3):
            run_full_correct()
        torch.cuda.synchronize()
        results["full_27_layers"] = triton.testing.do_bench(run_full_correct, warmup=10, rep=50)

        # 5. Full 27-layer pass WITHOUT shuffle (hypothetical fusion)
        def run_no_shuffle():
            # Pretend we fused shuffle into attention
            out = x_shuffled  # Start already shuffled
            for _ in range(depth):
                out = attn(out, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            return out  # Skip unshuffle

        for _ in range(3):
            run_no_shuffle()
        torch.cuda.synchronize()
        results["no_shuffle"] = triton.testing.do_bench(run_no_shuffle, warmup=10, rep=50)

        # Analysis
        shuffle_total = results["shuffle"] + results["unshuffle"]
        attn_total = depth * results["single_attn"]

        print(f"\n{'Component':<25} {'Time (ms)':>10}")
        print("-" * 40)
        print(f"{'Shuffle (1x)':<25} {results['shuffle']:>10.3f}")
        print(f"{'Unshuffle (1x)':<25} {results['unshuffle']:>10.3f}")
        print(f"{'Shuffle+Unshuffle total':<25} {shuffle_total:>10.3f}")
        print(f"{'Single attention layer':<25} {results['single_attn']:>10.3f}")
        print(f"{'Attention x27 (computed)':<25} {attn_total:>10.3f}")
        print("-" * 40)
        print(f"{'Full 27 layers (measured)':<25} {results['full_27_layers']:>10.3f}")
        print(f"{'Without shuffle (measured)':<25} {results['no_shuffle']:>10.3f}")

        # Overhead analysis
        shuffle_pct = 100 * shuffle_total / results["full_27_layers"]
        saved_if_fused = shuffle_total
        potential_speedup = results["full_27_layers"] / results["no_shuffle"]

        print(f"\n--- Analysis ---")
        print(f"Shuffle overhead:         {shuffle_total:.3f} ms ({shuffle_pct:.1f}% of total)")
        print(f"Potential speedup if fused: {potential_speedup:.2f}x")
        print(f"Time saved if fused:      {saved_if_fused:.3f} ms")

        # Is it worth fusing?
        threshold = 0.1  # 10% overhead threshold
        worth_it = shuffle_pct > threshold * 100
        print(f"\nWorth fusing? {'YES' if worth_it else 'NO'} (threshold: {threshold*100:.0f}%)")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    with app.run():
        run_v14.remote()
