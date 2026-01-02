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
def run_v12():
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
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    print("=" * 70)
    print("Windowed Attention Breakdown")
    print("=" * 70)

    from vision_attention import (
        TritonVisionAttention,
        compute_vision_rope_2d,
        _window_attention_rope_kernel,
        _window_attention_kernel,
        next_power_of_2,
    )
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    # Test sizes
    for img_size in [448, 896, 1344]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        t, h, w = grid_thw[0]
        seq_len = t * h * w
        merge_unit = spatial_merge_size ** 2

        print(f"\n{'='*70}")
        print(f"Image: {img_size}x{img_size} -> seq_len={seq_len}")
        print("=" * 70)

        # Get window indices
        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        num_windows = len(cu_seqlens) - 1

        print(f"Windows: {num_windows}, tokens per window: ~{seq_len // num_windows}")

        # Create test tensors
        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

        # Initialize module
        attn = TritonVisionAttention(
            hidden_size, num_heads, head_dim,
            spatial_merge_size=spatial_merge_size
        ).to(device, dtype)

        results = {}

        # 1. Window shuffle
        def run_shuffle():
            return window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(10):
            run_shuffle()
        torch.cuda.synchronize()
        results["shuffle"] = triton.testing.do_bench(run_shuffle, warmup=20, rep=100)

        x_shuffled = run_shuffle()

        # 2. QKV Projection
        def run_qkv():
            return attn.qkv(x_shuffled)

        for _ in range(10):
            run_qkv()
        torch.cuda.synchronize()
        results["qkv_proj"] = triton.testing.do_bench(run_qkv, warmup=20, rep=100)

        qkv = run_qkv()
        qkv_reshaped = qkv.reshape(seq_len, 3, num_heads, head_dim)
        q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]

        # 3. RoPE computation
        def run_rope_compute():
            return compute_vision_rope_2d(
                grid_thw,
                head_dim=head_dim,
                spatial_merge_size=spatial_merge_size,
                rope_theta=10000.0,
                device=device,
                dtype=dtype,
                window_idx=window_index,
            )

        for _ in range(10):
            run_rope_compute()
        torch.cuda.synchronize()
        results["rope_compute"] = triton.testing.do_bench(run_rope_compute, warmup=20, rep=100)

        cos, sin = run_rope_compute()

        # 4. Attention kernel only (with RoPE fused)
        out = torch.empty_like(q)
        max_ctx = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        padded_head_dim = next_power_of_2(head_dim)
        padded_half_dim = next_power_of_2(head_dim // 2)

        def run_attn_kernel():
            grid = lambda meta: (triton.cdiv(max_ctx, meta["BLOCK_M"]), num_windows * num_heads)
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
                num_windows, num_heads,
                max_ctx,
                HEAD_DIM=head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
                PADDED_HALF_DIM=padded_half_dim,
            )
            return out

        for _ in range(10):
            run_attn_kernel()
        torch.cuda.synchronize()
        results["attn_kernel"] = triton.testing.do_bench(run_attn_kernel, warmup=20, rep=100)

        # 5. Output projection
        out_flat = out.reshape(seq_len, hidden_size)

        def run_o_proj():
            return attn.proj(out_flat)

        for _ in range(10):
            run_o_proj()
        torch.cuda.synchronize()
        results["o_proj"] = triton.testing.do_bench(run_o_proj, warmup=20, rep=100)

        o = run_o_proj()

        # 6. Window unshuffle
        def run_unshuffle():
            return window_unshuffle(o, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(10):
            run_unshuffle()
        torch.cuda.synchronize()
        results["unshuffle"] = triton.testing.do_bench(run_unshuffle, warmup=20, rep=100)

        # 7. Full end-to-end (for comparison)
        def run_full():
            xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
            out = attn(xs, cu_seqlens=cu_seqlens, grid_thw=grid_thw, window_idx=window_index)
            return window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)

        for _ in range(10):
            run_full()
        torch.cuda.synchronize()
        results["full_e2e"] = triton.testing.do_bench(run_full, warmup=20, rep=100)

        # Print breakdown
        print(f"\n{'Component':<20} {'Time (ms)':>10} {'% of E2E':>10}")
        print("-" * 45)

        component_sum = sum([
            results["shuffle"],
            results["qkv_proj"],
            results["rope_compute"],
            results["attn_kernel"],
            results["o_proj"],
            results["unshuffle"],
        ])

        for name in ["shuffle", "qkv_proj", "rope_compute", "attn_kernel", "o_proj", "unshuffle"]:
            pct = 100 * results[name] / results["full_e2e"]
            print(f"{name:<20} {results[name]:>10.3f} {pct:>9.1f}%")

        print("-" * 45)
        print(f"{'Sum of parts':<20} {component_sum:>10.3f}")
        print(f"{'Full E2E':<20} {results['full_e2e']:>10.3f}")
        print(f"{'Overhead':<20} {results['full_e2e'] - component_sum:>10.3f}")

        # Fusion potential
        print(f"\n--- Fusion Analysis ---")
        shuffle_overhead = results["shuffle"] + results["unshuffle"]
        rope_overhead = results["rope_compute"]
        kernel_time = results["attn_kernel"]
        proj_time = results["qkv_proj"] + results["o_proj"]

        print(f"Shuffle/unshuffle:   {shuffle_overhead:.3f} ms ({100*shuffle_overhead/results['full_e2e']:.1f}%)")
        print(f"RoPE compute:        {rope_overhead:.3f} ms ({100*rope_overhead/results['full_e2e']:.1f}%)")
        print(f"Attention kernel:    {kernel_time:.3f} ms ({100*kernel_time/results['full_e2e']:.1f}%)")
        print(f"Projections:         {proj_time:.3f} ms ({100*proj_time/results['full_e2e']:.1f}%)")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    with app.run():
        run_v12.remote()
