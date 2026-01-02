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
def run_v17():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    device = torch.device("cuda")
    dtype = torch.float16  # flash_attn requires fp16 or bf16

    print("=" * 70)
    print("OUR TRITON KERNEL vs FLASH-ATTENTION-2")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Check if flash_attn is available (NGC image has it)
    try:
        from flash_attn import flash_attn_varlen_func, flash_attn_func
        HAS_FLASH = True
        import flash_attn
        print(f"flash_attn version: {flash_attn.__version__}")
    except ImportError:
        HAS_FLASH = False
        print("flash_attn NOT available - can't compare")
        return

    from vision_attention import (
        TritonVisionAttention,
        compute_vision_rope_2d,
        _window_attention_rope_kernel,
        next_power_of_2,
    )
    from window_attn import get_window_index, window_shuffle, window_unshuffle

    # Qwen3-VL vision config
    hidden_size = 1152
    num_heads = 16
    head_dim = 72  # Non-power-of-2!
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    print(f"\nConfig: heads={num_heads}, head_dim={head_dim}, hidden={hidden_size}")
    print(f"Note: head_dim=72 is non-power-of-2 (challenging for kernels)")
    print()

    results = []

    for img_size in [448, 672, 896, 1344]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        print(f"\n{'='*70}")
        print(f"Image: {img_size}x{img_size} | seq_len={seq_len}")
        print("=" * 70)

        # Get window info
        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        num_windows = len(cu_seqlens) - 1
        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        print(f"Windows: {num_windows}, max_seqlen_per_window: {max_seqlen}")

        # Create Q, K, V tensors [seq, heads, head_dim]
        q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Precompute RoPE
        cos, sin = compute_vision_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
            rope_theta=10000.0,
            device=device,
            dtype=dtype,
            window_idx=window_index,
        )

        # ============================================================
        # FLASH-ATTENTION-2 (varlen for windowed)
        # ============================================================
        def flash_attn_windowed():
            return flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=head_dim ** -0.5,
                causal=False,
            )

        # Warmup
        for _ in range(5):
            flash_attn_windowed()
        torch.cuda.synchronize()

        flash_ms = triton.testing.do_bench(flash_attn_windowed, warmup=20, rep=100)
        print(f"Flash-Attention-2 (varlen): {flash_ms:.3f} ms")

        # ============================================================
        # OUR TRITON KERNEL (with fused RoPE)
        # ============================================================
        out = torch.empty_like(q)
        padded_head_dim = next_power_of_2(head_dim)
        padded_half_dim = next_power_of_2(head_dim // 2)

        def our_kernel():
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

        for _ in range(5):
            our_kernel()
        torch.cuda.synchronize()

        our_ms = triton.testing.do_bench(our_kernel, warmup=20, rep=100)
        print(f"Our Triton kernel (fused RoPE): {our_ms:.3f} ms")

        # ============================================================
        # FLASH-ATTENTION-2 + SEPARATE ROPE (fairer comparison)
        # ============================================================
        def apply_rope_pytorch(x, cos, sin):
            """Apply RoPE in PyTorch"""
            half = head_dim // 2
            x0, x1 = x[..., :half], x[..., half:]
            cos = cos[:, None, :]  # [seq, 1, half]
            sin = sin[:, None, :]
            return torch.cat([
                x0 * cos - x1 * sin,
                x0 * sin + x1 * cos
            ], dim=-1)

        def flash_with_rope():
            q_rope = apply_rope_pytorch(q, cos, sin)
            k_rope = apply_rope_pytorch(k, cos, sin)
            return flash_attn_varlen_func(
                q_rope, k_rope, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=0.0,
                softmax_scale=head_dim ** -0.5,
                causal=False,
            )

        for _ in range(5):
            flash_with_rope()
        torch.cuda.synchronize()

        flash_rope_ms = triton.testing.do_bench(flash_with_rope, warmup=20, rep=100)
        print(f"Flash-Attention-2 + PyTorch RoPE: {flash_rope_ms:.3f} ms")

        # Results
        print(f"\n--- Comparison ---")
        print(f"Flash (no RoPE):     {flash_ms:.3f} ms")
        print(f"Flash + RoPE:        {flash_rope_ms:.3f} ms")
        print(f"Ours (fused RoPE):   {our_ms:.3f} ms")

        vs_flash = flash_ms / our_ms
        vs_flash_rope = flash_rope_ms / our_ms

        print(f"\nOurs vs Flash (no RoPE): {vs_flash:.2f}x {'FASTER' if vs_flash > 1 else 'SLOWER'}")
        print(f"Ours vs Flash+RoPE:      {vs_flash_rope:.2f}x {'FASTER' if vs_flash_rope > 1 else 'SLOWER'}")

        results.append({
            "img_size": img_size,
            "seq_len": seq_len,
            "flash_ms": flash_ms,
            "flash_rope_ms": flash_rope_ms,
            "our_ms": our_ms,
            "vs_flash": vs_flash,
            "vs_flash_rope": vs_flash_rope,
        })

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Image':<10} {'SeqLen':<8} {'Flash':<10} {'Flash+RoPE':<12} {'Ours':<10} {'vs Flash+RoPE':<15}")
    print("-" * 70)
    for r in results:
        status = "WIN" if r["vs_flash_rope"] > 1 else "LOSE"
        print(f"{r['img_size']:<10} {r['seq_len']:<8} {r['flash_ms']:<10.3f} {r['flash_rope_ms']:<12.3f} {r['our_ms']:<10.3f} {r['vs_flash_rope']:.2f}x {status}")

    print("\n" + "=" * 70)
    wins = sum(1 for r in results if r["vs_flash_rope"] > 1)
    print(f"VERDICT: {'WE WIN' if wins > len(results)/2 else 'FLASH WINS'} ({wins}/{len(results)} cases)")
    print("=" * 70)

    return results


if __name__ == "__main__":
    with app.run():
        run_v17.remote()
