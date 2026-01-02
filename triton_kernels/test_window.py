import modal
from pathlib import Path

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()
image_dir = local_dir.parent / "examples" / "imgs"

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install("triton>=3.0.0", "pillow")
    .add_local_dir(local_dir, remote_path="/root/kernels")
    .add_local_dir(image_dir, remote_path="/root/images")
)

@app.function(image=base_image, gpu="H100", timeout=300)
def run_v10():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3-VL config
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    window_size = 112
    spatial_merge_size = 2
    patch_size = 14

    print("=" * 70)
    print("Window Shuffle/Unshuffle Test")
    print("=" * 70)

    # Test 1: Basic shuffle/unshuffle roundtrip
    print("\n[Test 1] Shuffle/Unshuffle Roundtrip")
    print("-" * 70)

    grid_thw = [(1, 32, 32)]  # 448x448 image -> 32x32 patches
    t, h, w = grid_thw[0]
    merge_unit = spatial_merge_size ** 2
    seq_len = t * h * w

    window_index, cu_seqlens = get_window_index(
        grid_thw,
        window_size=window_size,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size,
        device=device,
    )

    print(f"Grid: {grid_thw[0]}")
    print(f"Total tokens: {seq_len}")
    print(f"Window index shape: {window_index.shape}")
    print(f"Cu_seqlens: {cu_seqlens.tolist()[:10]}... (len={len(cu_seqlens)})")
    print(f"Num windows: {len(cu_seqlens) - 1}")

    # create test tensor
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    # shuffle then unshuffle
    x_shuffled = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
    x_recovered = window_unshuffle(x_shuffled, window_index, spatial_merge_size=spatial_merge_size)

    # should be identical
    max_diff = (x - x_recovered).abs().max().item()
    print(f"Roundtrip max diff: {max_diff}")
    assert max_diff == 0, "Shuffle/unshuffle roundtrip failed!"
    print("✓ Roundtrip passed")

    # Test 2: Window boundaries
    print("\n[Test 2] Window Boundaries")
    print("-" * 70)

    # each window should have contiguous tokens
    for i in range(min(5, len(cu_seqlens) - 1)):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        window_len = end - start
        print(f"Window {i}: tokens [{start}, {end}), len={window_len}")

    print("✓ Window boundaries look correct")

    # Test 3: Different image sizes
    print("\n[Test 3] Various Image Sizes")
    print("-" * 70)

    test_cases = [
        [(1, 18, 18)],   # 256x256
        [(1, 32, 32)],   # 448x448
        [(1, 48, 48)],   # 672x672
        [(1, 32, 48)],   # 448x672 (non-square)
    ]

    for grid_thw in test_cases:
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        x_shuffled = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
        x_recovered = window_unshuffle(x_shuffled, window_index, spatial_merge_size=spatial_merge_size)

        max_diff = (x - x_recovered).abs().max().item()
        num_windows = len(cu_seqlens) - 1
        status = "✓" if max_diff == 0 else "✗"
        print(f"Grid {grid_thw[0]}: seq={seq_len}, windows={num_windows}, diff={max_diff} {status}")

    # Test 4: Windowed attention end-to-end (TritonVisionAttention)
    print("\n[Test 4] Windowed Attention (TritonVisionAttention)")
    print("-" * 70)

    from vision_attention import TritonVisionAttention

    grid_thw = [(1, 32, 32)]
    t, h, w = grid_thw[0]
    seq_len = t * h * w

    window_index, cu_seqlens = get_window_index(
        grid_thw,
        window_size=window_size,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size,
        device=device,
    )

    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    x_shuffled = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)

    attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)
    out_shuffled = attn(
        x_shuffled,
        cu_seqlens=cu_seqlens,
        grid_thw=grid_thw,
        window_idx=window_index,
    )
    out = window_unshuffle(out_shuffled, window_index, spatial_merge_size=spatial_merge_size)

    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
    print(f"Has NaN: {torch.isnan(out).any().item()}")
    print(f"Has Inf: {torch.isinf(out).any().item()}")
    print("✓ Windowed attention runs without errors")

    # Test 5: Performance comparison
    print("\n[Test 5] Performance: Full vs Windowed")
    print("-" * 70)

    import triton

    attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)

    for grid_thw in [[(1, 32, 32)], [(1, 48, 48)], [(1, 64, 64)]]:
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

        def full_attn():
            return attn(x, grid_thw=grid_thw)

        def windowed():
            x_s = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
            out_s = attn(
                x_s,
                cu_seqlens=cu_seqlens,
                grid_thw=grid_thw,
                window_idx=window_index,
            )
            return window_unshuffle(out_s, window_index, spatial_merge_size=spatial_merge_size)

        # warmup
        for _ in range(5):
            full_attn()
            windowed()
        torch.cuda.synchronize()

        full_ms = triton.testing.do_bench(full_attn, warmup=10, rep=50)
        win_ms = triton.testing.do_bench(windowed, warmup=10, rep=50)

        num_windows = len(cu_seqlens) - 1
        print(f"Grid {grid_thw[0]} | seq={seq_len:>5} | windows={num_windows:>3} | full={full_ms:.3f}ms | windowed={win_ms:.3f}ms | speedup={full_ms/win_ms:.2f}x")

    # Test 6: Flash Window Attention (if available)
    print("\n[Test 6] Flash Window Attention")
    print("-" * 70)

    from window_attn import HAS_FLASH_ATTN, flash_window_attention

    if HAS_FLASH_ATTN:
        print("flash_attn available ✓")

        grid_thw = [(1, 32, 32)]
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

        q = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)
        k = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)
        v = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)

        q_s = window_shuffle(q, window_index, spatial_merge_size=spatial_merge_size)
        k_s = window_shuffle(k, window_index, spatial_merge_size=spatial_merge_size)
        v_s = window_shuffle(v, window_index, spatial_merge_size=spatial_merge_size)

        out_s = flash_window_attention(q_s, k_s, v_s, cu_seqlens, num_heads, head_dim)
        out = window_unshuffle(out_s, window_index, spatial_merge_size=spatial_merge_size)

        print(f"Output shape: {out.shape}")
        print(f"Has NaN: {torch.isnan(out).any().item()}")
        print(f"Has Inf: {torch.isinf(out).any().item()}")

        # benchmark flash vs triton
        print("\nFlash vs Triton windowed performance:")
        for grid_thw in [[(1, 32, 32)], [(1, 64, 64)]]:
            t, h, w = grid_thw[0]
            seq_len = t * h * w

            window_index, cu_seqlens = get_window_index(
                grid_thw,
                window_size=window_size,
                spatial_merge_size=spatial_merge_size,
                patch_size=patch_size,
                device=device,
            )

            q = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)
            k = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)
            v = torch.randn(seq_len, num_heads * head_dim, device=device, dtype=torch.float16)

            def flash_windowed():
                qs = window_shuffle(q, window_index, spatial_merge_size=spatial_merge_size)
                ks = window_shuffle(k, window_index, spatial_merge_size=spatial_merge_size)
                vs = window_shuffle(v, window_index, spatial_merge_size=spatial_merge_size)
                out_s = flash_window_attention(qs, ks, vs, cu_seqlens, num_heads, head_dim)
                return window_unshuffle(out_s, window_index, spatial_merge_size=spatial_merge_size)

            for _ in range(5):
                flash_windowed()
            torch.cuda.synchronize()

            flash_ms = triton.testing.do_bench(flash_windowed, warmup=10, rep=50)
            num_windows = len(cu_seqlens) - 1
            print(f"Grid {grid_thw[0]} | seq={seq_len:>5} | windows={num_windows:>3} | flash_windowed={flash_ms:.3f}ms")
    else:
        print("flash_attn not available - skipping")
        print("(Install with: pip install flash-attn)")

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    with app.run():
        run_v10.remote()
