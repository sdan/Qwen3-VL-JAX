"""Generate the one chart: Flash+RoPE vs fused kernel.

Run (from repo root): modal run triton_kernels/bench_chart.py
Run (from triton_kernels/): modal run bench_chart.py

Outputs:
- ASCII bar chart (terminal)
- CSV data (for matplotlib/plotting)
- The story: "We fused RoPE into attention. X× faster."

Note: This is a per-layer attention microbenchmark (it does not include MLPs, QKV/proj GEMMs, etc.).
"""

import modal
from pathlib import Path

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install("triton>=3.0.0")
    .add_local_dir(local_dir, remote_path="/root/kernels")
)


def ascii_bar_chart(results: list[dict]) -> str:
    """Generate ASCII bar chart for HN/blog."""

    max_time = max(r["flash_rope_ms"] for r in results)
    chart_width = 40

    lines = []
    lines.append("")
    lines.append("  Time (ms) — Lower is Better")
    lines.append("  " + "─" * 50)
    lines.append("")

    for r in results:
        img = f"{r['img_size']}×{r['img_size']}"

        # Flash+RoPE bar
        flash_len = int((r["flash_rope_ms"] / max_time) * chart_width)
        flash_bar = "█" * flash_len
        lines.append(f"  {img:<11} Flash+RoPE │{flash_bar} {r['flash_rope_ms']:.3f}ms")

        # Fused bar
        fused_len = int((r["our_ms"] / max_time) * chart_width)
        fused_bar = "░" * fused_len
        speedup = r["vs_flash_rope"]
        lines.append(f"          Fused      │{fused_bar} {r['our_ms']:.3f}ms  ({speedup:.1f}x faster)")
        lines.append("")

    lines.append("  " + "─" * 50)
    lines.append("")

    # Add the insight
    avg_speedup = sum(r["vs_flash_rope"] for r in results) / len(results)
    lines.append(f"  Average speedup: {avg_speedup:.1f}x")
    lines.append("  By eliminating one memory round-trip.")
    lines.append("")

    return "\n".join(lines)


def csv_output(results: list[dict]) -> str:
    """Generate CSV for plotting tools."""
    lines = ["image_size,seq_len,flash_rope_ms,fused_ms,speedup"]
    for r in results:
        lines.append(f"{r['img_size']},{r['seq_len']},{r['flash_rope_ms']:.4f},{r['our_ms']:.4f},{r['vs_flash_rope']:.2f}")
    return "\n".join(lines)


def markdown_table(results: list[dict]) -> str:
    """Generate markdown table for blog/README."""
    lines = []
    lines.append("| Image Size | Seq Length | Flash+RoPE | Fused | Speedup |")
    lines.append("|------------|------------|------------|-------|---------|")
    for r in results:
        lines.append(f"| {r['img_size']}×{r['img_size']} | {r['seq_len']:,} | {r['flash_rope_ms']:.3f} ms | {r['our_ms']:.3f} ms | **{r['vs_flash_rope']:.1f}×** |")
    return "\n".join(lines)


@app.function(image=base_image, gpu="H100", timeout=600)
def run_benchmark():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    device = torch.device("cuda")
    dtype = torch.float16

    # Check flash_attn
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        print("ERROR: flash_attn not available")
        return None

    from vision_attention import (
        compute_vision_rope_2d,
        _window_attention_rope_kernel,
        next_power_of_2,
    )
    from window_attn import get_window_index

    # Qwen3-VL config
    num_heads = 16
    head_dim = 72
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    print("=" * 60)
    print("BENCHMARK: Flash+RoPE vs Fused Kernel")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Config: heads={num_heads}, head_dim={head_dim}")
    print()

    results = []

    for img_size in [448, 672, 896, 1344]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        seq_len = patches_per_side * patches_per_side

        # Window setup
        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        num_windows = len(cu_seqlens) - 1
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        # Tensors
        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)

        cos, sin = compute_vision_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
            rope_theta=10000.0,
            device=device,
            dtype=dtype,
            window_idx=window_index,
        )

        # --- Flash + RoPE (baseline) ---
        def apply_rope(x, cos, sin):
            half = head_dim // 2
            x0, x1 = x[..., :half], x[..., half:]
            cos_exp = cos[:, None, :]
            sin_exp = sin[:, None, :]
            return torch.cat([x0 * cos_exp - x1 * sin_exp, x0 * sin_exp + x1 * cos_exp], dim=-1)

        def flash_rope():
            q_rot = apply_rope(q, cos, sin)
            k_rot = apply_rope(k, cos, sin)
            return flash_attn_varlen_func(
                q_rot, k_rot, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=head_dim ** -0.5,
                causal=False,
            )

        # --- Our Fused Kernel ---
        out = torch.empty_like(q)
        padded_head_dim = next_power_of_2(head_dim)
        padded_half_dim = next_power_of_2(head_dim // 2)

        def fused_kernel():
            grid = lambda meta: (triton.cdiv(max_seqlen, meta["BLOCK_M"]), num_windows * num_heads)
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
                max_seqlen,
                HEAD_DIM=head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
                PADDED_HALF_DIM=padded_half_dim,
            )
            return out

        # Warmup
        for _ in range(10):
            flash_rope()
            fused_kernel()
        torch.cuda.synchronize()

        # Benchmark
        flash_rope_ms = triton.testing.do_bench(flash_rope, warmup=20, rep=100)
        our_ms = triton.testing.do_bench(fused_kernel, warmup=20, rep=100)
        speedup = flash_rope_ms / our_ms

        print(f"{img_size}×{img_size}: Flash+RoPE={flash_rope_ms:.3f}ms, Fused={our_ms:.3f}ms, {speedup:.1f}x faster")

        results.append({
            "img_size": img_size,
            "seq_len": seq_len,
            "flash_rope_ms": flash_rope_ms,
            "our_ms": our_ms,
            "vs_flash_rope": speedup,
        })

    return results


@app.local_entrypoint()
def main():
    results = run_benchmark.remote()

    if results is None:
        print("Benchmark failed")
        return

    # Print ASCII chart
    print("\n" + "=" * 60)
    print("THE CHART")
    print("=" * 60)
    print(ascii_bar_chart(results))

    # Print markdown table
    print("\n" + "=" * 60)
    print("MARKDOWN TABLE (for blog)")
    print("=" * 60)
    print(markdown_table(results))

    # Print CSV
    print("\n" + "=" * 60)
    print("CSV DATA (for plotting)")
    print("=" * 60)
    print(csv_output(results))

    # Save CSV to file
    csv_path = Path(__file__).parent / "benchmark_results.csv"
    csv_path.write_text(csv_output(results))
    print(f"\nSaved to: {csv_path}")
