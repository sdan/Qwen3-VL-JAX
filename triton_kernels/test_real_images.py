import modal
from pathlib import Path

app = modal.App("accel")

# Paths relative to script location
script_dir = Path(__file__).parent.resolve()
local_dir = script_dir  # triton_kernels directory
image_dir = script_dir.parent / "examples" / "imgs"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "triton>=3.0.0", "pillow", "numpy")
    .add_local_dir(local_dir, remote_path="/root/kernels")
    .add_local_dir(image_dir, remote_path="/root/images")
)

@app.function(image=base_image, gpu="H100", timeout=300)
def run_v1():
    import sys
    sys.path.insert(0, "/root/kernels")

    import os
    import math
    import torch
    import numpy as np
    from PIL import Image
    from vision_attention import TritonVisionAttention, UnfusedVisionAttention
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3-VL vision config
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    patch_size = 14
    spatial_merge_size = 2
    temporal_patch_size = 2
    window_size = 112

    # Image preprocessing constants (matches Qwen3-VL)
    IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def smart_resize(height: int, width: int, factor: int,
                     min_pixels: int = 56*56, max_pixels: int = 12845056) -> tuple[int, int]:
        """Resize dimensions to align with patch grid"""
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)

        area = h_bar * w_bar
        if area > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif area < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = max(factor, math.ceil(height * beta / factor) * factor)
            w_bar = max(factor, math.ceil(width * beta / factor) * factor)

        return int(h_bar), int(w_bar)

    def preprocess_image_torch(image_path: str) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        """Convert image to Qwen3-VL format (PyTorch version for GPU testing)"""
        pil_img = Image.open(image_path).convert("RGB")
        width, height = pil_img.size

        factor = patch_size * spatial_merge_size  # 28
        new_h, new_w = smart_resize(height, width, factor)

        if (new_w, new_h) != (width, height):
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        # Normalize to [-1, 1]
        image_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGE_MEAN) / IMAGE_STD

        # CHW + temporal axis: (1, C, H, W)
        image_np = np.transpose(image_np, (2, 0, 1))[None, ...]

        # Pad temporal dimension if needed
        frames = image_np.shape[0]
        if temporal_patch_size > 1 and frames % temporal_patch_size != 0:
            pad = temporal_patch_size - (frames % temporal_patch_size)
            image_np = np.concatenate([image_np, np.repeat(image_np[-1:], pad, axis=0)], axis=0)

        frames, channels, new_h, new_w = image_np.shape
        grid_t = frames // temporal_patch_size
        grid_h = new_h // patch_size
        grid_w = new_w // patch_size

        # Total vision tokens = grid_t * grid_h * grid_w
        total_tokens = grid_t * grid_h * grid_w

        return total_tokens, [(grid_t, grid_h, grid_w)]

    print("=" * 70)
    print("Testing Triton Vision Attention with REAL IMAGES")
    print("=" * 70)
    print(f"Config: hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"Patch: size={patch_size}, spatial_merge={spatial_merge_size}")
    print()

    # Get all test images
    image_dir_path = "/root/images"
    test_images = [f for f in os.listdir(image_dir_path)
                   if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(test_images)} test images")
    print("-" * 70)

    # Create attention modules
    fused = TritonVisionAttention(
        hidden_size, num_heads, head_dim,
        spatial_merge_size=spatial_merge_size
    ).to(device, dtype)

    unfused = UnfusedVisionAttention(
        hidden_size, num_heads, head_dim,
        spatial_merge_size=spatial_merge_size
    ).to(device, dtype)
    unfused.load_state_dict(fused.state_dict(), strict=True)

    results = []

    for img_name in test_images:
        img_path = os.path.join(image_dir_path, img_name)

        try:
            # Preprocess image to get dimensions
            total_tokens, grid_thw = preprocess_image_torch(img_path)
            grid_t, grid_h, grid_w = grid_thw[0]

            # Get original image dimensions
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size

            print(f"\n{img_name}:")
            print(f"  Original: {orig_w}x{orig_h}")
            print(f"  Grid (t,h,w): ({grid_t}, {grid_h}, {grid_w})")
            print(f"  Total tokens: {total_tokens}")

            # Create realistic input (simulating output from patch embedding)
            # In real model, this would be pixel_values -> patch_embed -> LayerNorm
            x = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

            # Qwen3-VL window shuffle (groups tokens per window contiguously)
            window_idx, cu_window = get_window_index(
                grid_thw,
                window_size=window_size,
                spatial_merge_size=spatial_merge_size,
                patch_size=patch_size,
                device=device,
            )
            assert cu_window[0].item() == 0 and cu_window[-1].item() == total_tokens

            # Shuffle into window order, run window attention, then unshuffle back.
            x_win = window_shuffle(x, window_idx, spatial_merge_size=spatial_merge_size)

            # Run with 2D RoPE (proper vision positional encoding) + window attention
            out_fused_win = fused(
                x_win,
                cu_seqlens=cu_window,
                grid_thw=grid_thw,
                window_idx=window_idx,
            )
            out_unfused_win = unfused(
                x_win,
                cu_seqlens=cu_window,
                grid_thw=grid_thw,
                window_idx=window_idx,
            )
            out_fused = window_unshuffle(
                out_fused_win, window_idx, spatial_merge_size=spatial_merge_size
            )
            out_unfused = window_unshuffle(
                out_unfused_win, window_idx, spatial_merge_size=spatial_merge_size
            )

            # Check correctness
            diff = (out_fused - out_unfused).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # Statistical comparison (outputs should have similar distributions)
            fused_mean = out_fused.mean().item()
            unfused_mean = out_unfused.mean().item()
            fused_std = out_fused.std().item()
            unfused_std = out_unfused.std().item()

            # Statistical validation:
            # - For std: use relative diff (both should be similar scale)
            # - For mean: use absolute diff (both near zero is fine)
            mean_abs_diff = abs(fused_mean - unfused_mean)
            std_rel_diff = abs(fused_std - unfused_std) / (abs(unfused_std) + 1e-6)

            # Also run with 1D RoPE for comparison
            out_fused_1d_win = fused(x_win, cu_seqlens=cu_window)  # No grid_thw = 1D RoPE
            out_fused_1d = window_unshuffle(
                out_fused_1d_win, window_idx, spatial_merge_size=spatial_merge_size
            )

            # 2D vs 1D should be different (different positional encoding)
            diff_2d_1d = (out_fused - out_fused_1d).abs().mean().item()

            # bf16 FlashAttention tolerance:
            # - max_diff < 0.25 is typical for bf16 online softmax vs standard
            # - mean_abs_diff < 0.01 (both should center near same value)
            # - std_rel_diff < 0.1 (variance should be similar)
            numerically_ok = max_diff < 0.25 and mean_abs_diff < 0.01 and std_rel_diff < 0.1
            status = "✓" if numerically_ok else "✗"

            print(f"  Fused vs Unfused: max={max_diff:.4f}, mean={mean_diff:.4f} {status}")
            print(f"  Stats - mean: {fused_mean:.4f} vs {unfused_mean:.4f} (abs_diff={mean_abs_diff:.6f})")
            print(f"  Stats - std:  {fused_std:.4f} vs {unfused_std:.4f} (rel_diff={std_rel_diff:.4f})")
            print(f"  2D vs 1D RoPE diff: {diff_2d_1d:.6f}")

            # Check for numerical issues
            has_nan = torch.isnan(out_fused).any().item()
            has_inf = torch.isinf(out_fused).any().item()

            if has_nan or has_inf:
                print(f"  ⚠ WARNING: NaN={has_nan}, Inf={has_inf}")

            results.append({
                "image": img_name,
                "tokens": total_tokens,
                "grid": grid_thw[0],
                "max_diff": max_diff,
                "mean_abs_diff": mean_abs_diff,
                "std_rel_diff": std_rel_diff,
                "passed": numerically_ok and not has_nan and not has_inf
            })

        except Exception as e:
            print(f"\n{img_name}: ERROR - {e}")
            results.append({"image": img_name, "passed": False, "error": str(e)})

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results if r.get("passed", False))
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All real image tests PASSED!")
    else:
        print("\n✗ Some tests failed:")
        for r in results:
            if not r.get("passed", False):
                print(f"  - {r['image']}: {r.get('error', 'numerical mismatch')}")

    # Test with typical production image sizes (simulated)
    print("\n" + "=" * 70)
    print("SCALING TEST (typical production sizes)")
    print("=" * 70)

    production_sizes = [
        ("448x448 (Qwen3-VL default)", 448, 448),
        ("672x672 (high-res)", 672, 672),
        ("896x896 (max typical)", 896, 896),
        ("1024x768 (landscape)", 1024, 768),
    ]

    for name, w, h in production_sizes:
        factor = patch_size * spatial_merge_size  # 28
        new_h = max(factor, round(h / factor) * factor)
        new_w = max(factor, round(w / factor) * factor)

        grid_h = new_h // patch_size
        grid_w = new_w // patch_size
        total_tokens = 1 * grid_h * grid_w
        grid_thw = [(1, grid_h, grid_w)]

        x = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)

        window_idx, cu_window = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        x_win = window_shuffle(x, window_idx, spatial_merge_size=spatial_merge_size)
        out_fused = window_unshuffle(
            fused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx),
            window_idx,
            spatial_merge_size=spatial_merge_size,
        )
        out_unfused = window_unshuffle(
            unfused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx),
            window_idx,
            spatial_merge_size=spatial_merge_size,
        )

        max_diff = (out_fused - out_unfused).abs().max().item()
        has_nan = torch.isnan(out_fused).any().item()
        has_inf = torch.isinf(out_fused).any().item()

        status = "✓" if max_diff < 0.25 and not has_nan and not has_inf else "✗"
        print(f"{name}: grid=({grid_h},{grid_w}), tokens={total_tokens}, max_diff={max_diff:.4f} {status}")

    # Benchmark with 448x448 (Qwen3-VL default) - more realistic
    print("\n" + "=" * 70)
    print("PERFORMANCE BENCHMARK (448x448 default size)")
    print("=" * 70)

    # 448x448 -> 32x32 grid -> 1024 tokens
    factor = patch_size * spatial_merge_size
    bench_h = bench_w = 448
    bench_h = max(factor, round(bench_h / factor) * factor)
    bench_w = max(factor, round(bench_w / factor) * factor)
    grid_h = bench_h // patch_size
    grid_w = bench_w // patch_size
    tokens = grid_h * grid_w
    grid_thw = [(1, grid_h, grid_w)]

    x = torch.randn(tokens, hidden_size, device=device, dtype=dtype)
    window_idx, cu_window = get_window_index(
        grid_thw,
        window_size=window_size,
        spatial_merge_size=spatial_merge_size,
        patch_size=patch_size,
        device=device,
    )
    x_win = window_shuffle(x, window_idx, spatial_merge_size=spatial_merge_size)

    # Warmup
    for _ in range(10):
        _ = fused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx)
        _ = unfused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx)
    torch.cuda.synchronize()

    import triton
    fused_ms = triton.testing.do_bench(
        lambda: fused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx),
        warmup=20,
        rep=100,
    )
    unfused_ms = triton.testing.do_bench(
        lambda: unfused(x_win, cu_seqlens=cu_window, grid_thw=grid_thw, window_idx=window_idx),
        warmup=20,
        rep=100,
    )

    speedup = unfused_ms / fused_ms

    print(f"Image: 448x448 ({bench_h}x{bench_w} after resize)")
    print(f"Grid: {grid_h}x{grid_w} = {tokens} tokens")
    print(f"Fused:   {fused_ms:.3f} ms")
    print(f"Unfused: {unfused_ms:.3f} ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Throughput: {tokens / (fused_ms / 1000):,.0f} tokens/sec")

    return results

if __name__ == "__main__":
    with app.run():
        run_v1.remote()
