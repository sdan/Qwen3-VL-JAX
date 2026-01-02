import modal
from pathlib import Path

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()

base_image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.01-py3")
    .pip_install(
        "triton>=3.0.0",
        "transformers>=4.45.0",
        "accelerate",
        "qwen-vl-utils",
    )
    .add_local_dir(local_dir, remote_path="/root/kernels")
)

@app.function(image=base_image, gpu="H100", timeout=900)
def run_v15():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton
    import time

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("Qwen3-VL Vision Encoder: Our Kernels vs HuggingFace")
    print("=" * 70)

    # ================================================================
    # Load HuggingFace model (vision encoder only)
    # ================================================================
    print("\n[1] Loading HuggingFace Qwen3-VL-2B vision encoder...")

    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

    # Use Qwen2-VL as proxy (Qwen3-VL may not be in transformers yet)
    # The vision encoder architecture is nearly identical
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    try:
        hf_model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda",
        )
        hf_visual = hf_model.visual
        print(f"Loaded: {model_name}")
        print(f"Vision encoder depth: {len(hf_visual.blocks)}")
        print(f"Hidden size: {hf_visual.embed_dim}")
    except Exception as e:
        print(f"Could not load HF model: {e}")
        print("Falling back to synthetic comparison...")
        hf_visual = None

    # ================================================================
    # Our optimized modules
    # ================================================================
    print("\n[2] Setting up our optimized kernels...")

    from vision_attention import TritonVisionAttention, compute_vision_rope_2d
    from window_attn import get_window_index, window_shuffle, window_unshuffle
    from fused_ops import swiglu, fused_add_rms_norm, layer_norm

    # Qwen3-VL vision config
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    intermediate_size = 4304
    depth = 27
    spatial_merge_size = 2
    window_size = 112
    patch_size = 14

    # Our attention module
    our_attn = TritonVisionAttention(
        hidden_size, num_heads, head_dim,
        spatial_merge_size=spatial_merge_size
    ).to(device, dtype)

    # MLP components
    gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False).to(device, dtype)
    up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False).to(device, dtype)
    down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False).to(device, dtype)

    # Norms
    ln_weight = torch.ones(hidden_size, device=device, dtype=dtype)
    ln_bias = torch.zeros(hidden_size, device=device, dtype=dtype)
    rms_weight = torch.ones(hidden_size, device=device, dtype=dtype)

    def our_block_forward(x, cos, sin, cu_seqlens):
        """Single transformer block with our fused ops"""
        # Pre-norm + attention
        normed = layer_norm(x, ln_weight, ln_bias)
        attn_out = our_attn(normed, cu_seqlens=cu_seqlens, cos=cos, sin=sin)
        x, residual = fused_add_rms_norm(attn_out, x, rms_weight)

        # MLP with fused SwiGLU
        gate = gate_proj(x)
        up = up_proj(x)
        mlp_out = down_proj(swiglu(gate, up))
        x = x + mlp_out

        return x

    # ================================================================
    # Benchmark
    # ================================================================
    print("\n[3] Benchmarking...")

    results = []

    for img_size in [448, 672, 896]:
        patches_per_side = img_size // patch_size
        grid_thw = [(1, patches_per_side, patches_per_side)]
        t, h, w = grid_thw[0]
        seq_len = t * h * w

        print(f"\n--- Image: {img_size}x{img_size} (seq_len={seq_len}) ---")

        # Get window indices
        window_index, cu_seqlens = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )

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

        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

        # ============================================================
        # Our optimized forward (27 layers)
        # ============================================================
        def our_forward():
            xs = window_shuffle(x, window_index, spatial_merge_size=spatial_merge_size)
            out = xs
            for _ in range(depth):
                out = our_block_forward(out, cos, sin, cu_seqlens)
            return window_unshuffle(out, window_index, spatial_merge_size=spatial_merge_size)

        # Warmup
        for _ in range(3):
            our_forward()
        torch.cuda.synchronize()

        our_ms = triton.testing.do_bench(our_forward, warmup=10, rep=50)
        print(f"Our kernels (27 layers): {our_ms:.2f} ms")

        # ============================================================
        # HuggingFace forward (if available)
        # ============================================================
        hf_ms = None
        if hf_visual is not None:
            # Create dummy pixel values matching the image size
            # HF expects [batch, channels, frames, height, width] for video
            # or [batch, channels, height, width] for image
            pixel_values = torch.randn(1, 3, img_size, img_size, device=device, dtype=dtype)
            grid_thw_tensor = torch.tensor([[1, patches_per_side, patches_per_side]], device=device)

            def hf_forward():
                with torch.no_grad():
                    return hf_visual(pixel_values, grid_thw=grid_thw_tensor)

            try:
                # Warmup
                for _ in range(3):
                    hf_forward()
                torch.cuda.synchronize()

                hf_ms = triton.testing.do_bench(hf_forward, warmup=10, rep=50)
                print(f"HuggingFace (full encoder): {hf_ms:.2f} ms")

                speedup = hf_ms / our_ms
                print(f"Speedup: {speedup:.2f}x")
            except Exception as e:
                print(f"HF benchmark failed: {e}")
                hf_ms = None

        results.append({
            "img_size": img_size,
            "seq_len": seq_len,
            "our_ms": our_ms,
            "hf_ms": hf_ms,
        })

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Image':<12} {'Seq Len':<10} {'Ours (ms)':<12} {'HF (ms)':<12} {'Speedup':<10}")
    print("-" * 60)
    for r in results:
        hf_str = f"{r['hf_ms']:.2f}" if r['hf_ms'] else "N/A"
        speedup_str = f"{r['hf_ms']/r['our_ms']:.2f}x" if r['hf_ms'] else "N/A"
        print(f"{r['img_size']}x{r['img_size']:<6} {r['seq_len']:<10} {r['our_ms']:<12.2f} {hf_str:<12} {speedup_str:<10}")

    print("\n" + "=" * 70)
    print("Note: HF includes patch embedding + merger; ours is attention+MLP only")
    print("=" * 70)

    return results


if __name__ == "__main__":
    with app.run():
        run_v15.remote()
