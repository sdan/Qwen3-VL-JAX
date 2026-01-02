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
def run_v11():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3-VL vision encoder config (same across 2B/8B/32B/72B)
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    intermediate_size = 4304
    depth = 27
    patch_size = 16
    spatial_merge_size = 2
    out_hidden_size = 4096  # for 8B model

    print("=" * 70)
    print("Qwen3-VL Vision Encoder Profiling")
    print("=" * 70)
    print(f"Config: hidden={hidden_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"        depth={depth}, intermediate={intermediate_size}")
    print(f"        patch={patch_size}, spatial_merge={spatial_merge_size}")
    print()

    # Test different image sizes
    image_sizes = [
        (448, 448),    # 28x28 patches -> 784 tokens -> 196 merged
        (672, 672),    # 42x42 patches -> 1764 tokens -> 441 merged
        (896, 896),    # 56x56 patches -> 3136 tokens -> 784 merged
        (1344, 1344),  # 84x84 patches -> 7056 tokens -> 1764 merged
    ]

    for img_h, img_w in image_sizes:
        patches_h = img_h // patch_size
        patches_w = img_w // patch_size
        num_patches = patches_h * patches_w
        merged_tokens = num_patches // (spatial_merge_size ** 2)

        print(f"\n{'='*70}")
        print(f"Image: {img_h}x{img_w} -> {patches_h}x{patches_w} patches -> {merged_tokens} merged tokens")
        print("=" * 70)

        # Simulate each component of vision encoder
        results = {}

        # 1. Patch Embedding (Conv2D)
        patch_embed = torch.nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size).to(device, dtype)
        img = torch.randn(1, 3, img_h, img_w, device=device, dtype=dtype)

        def run_patch_embed():
            return patch_embed(img)

        for _ in range(10):
            run_patch_embed()
        torch.cuda.synchronize()
        results["patch_embed"] = triton.testing.do_bench(run_patch_embed, warmup=20, rep=100)

        # 2. QKV Projection (per layer)
        x = torch.randn(num_patches, hidden_size, device=device, dtype=dtype)
        qkv_proj = torch.nn.Linear(hidden_size, 3 * num_heads * head_dim, bias=True).to(device, dtype)

        def run_qkv():
            return qkv_proj(x)

        for _ in range(10):
            run_qkv()
        torch.cuda.synchronize()
        results["qkv_proj"] = triton.testing.do_bench(run_qkv, warmup=20, rep=100)

        # 3. Attention (using our Triton kernel)
        from vision_attention import TritonVisionAttention

        attn = TritonVisionAttention(hidden_size, num_heads, head_dim, spatial_merge_size=spatial_merge_size).to(device, dtype)
        grid_thw = [(1, patches_h, patches_w)]

        def run_attn():
            return attn(x, grid_thw=grid_thw)

        for _ in range(10):
            run_attn()
        torch.cuda.synchronize()
        results["attention"] = triton.testing.do_bench(run_attn, warmup=20, rep=100)

        # 4. Output Projection (per layer)
        o_proj = torch.nn.Linear(num_heads * head_dim, hidden_size, bias=True).to(device, dtype)

        def run_o_proj():
            return o_proj(x)

        for _ in range(10):
            run_o_proj()
        torch.cuda.synchronize()
        results["o_proj"] = triton.testing.do_bench(run_o_proj, warmup=20, rep=100)

        # 5. MLP: gate + up + SwiGLU + down
        gate_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False).to(device, dtype)
        up_proj = torch.nn.Linear(hidden_size, intermediate_size, bias=False).to(device, dtype)
        down_proj = torch.nn.Linear(intermediate_size, hidden_size, bias=False).to(device, dtype)

        def run_mlp_pytorch():
            gate = gate_proj(x)
            up = up_proj(x)
            act = torch.nn.functional.silu(gate) * up
            return down_proj(act)

        for _ in range(10):
            run_mlp_pytorch()
        torch.cuda.synchronize()
        results["mlp_unfused"] = triton.testing.do_bench(run_mlp_pytorch, warmup=20, rep=100)

        # 5b. MLP with fused SwiGLU
        from fused_ops import swiglu

        def run_mlp_fused():
            gate = gate_proj(x)
            up = up_proj(x)
            act = swiglu(gate, up)
            return down_proj(act)

        for _ in range(10):
            run_mlp_fused()
        torch.cuda.synchronize()
        results["mlp_fused"] = triton.testing.do_bench(run_mlp_fused, warmup=20, rep=100)

        # 6. LayerNorm
        ln = torch.nn.LayerNorm(hidden_size).to(device, dtype)

        def run_ln():
            return ln(x)

        for _ in range(10):
            run_ln()
        torch.cuda.synchronize()
        results["layer_norm"] = triton.testing.do_bench(run_ln, warmup=20, rep=100)

        # 6b. Fused LayerNorm
        from fused_ops import layer_norm
        weight = torch.ones(hidden_size, device=device, dtype=dtype)
        bias = torch.zeros(hidden_size, device=device, dtype=dtype)

        def run_ln_fused():
            return layer_norm(x, weight, bias)

        for _ in range(10):
            run_ln_fused()
        torch.cuda.synchronize()
        results["layer_norm_fused"] = triton.testing.do_bench(run_ln_fused, warmup=20, rep=100)

        # 7. RMSNorm + Add (fused)
        from fused_ops import fused_add_rms_norm
        residual = torch.randn_like(x)
        rms_weight = torch.ones(hidden_size, device=device, dtype=dtype)

        def run_add_rms_unfused():
            s = x + residual
            var = s.float().pow(2).mean(-1, keepdim=True)
            return (s * torch.rsqrt(var + 1e-6)) * rms_weight, s

        def run_add_rms_fused():
            return fused_add_rms_norm(x, residual, rms_weight)

        for _ in range(10):
            run_add_rms_unfused()
            run_add_rms_fused()
        torch.cuda.synchronize()
        results["add_rms_unfused"] = triton.testing.do_bench(run_add_rms_unfused, warmup=20, rep=100)
        results["add_rms_fused"] = triton.testing.do_bench(run_add_rms_fused, warmup=20, rep=100)

        # 8. Spatial Merge (2x2)
        from spatial_merge import spatial_merge_2x2
        x_pre_merge = torch.randn(patches_h * patches_w, hidden_size, device=device, dtype=dtype)

        def run_merge():
            return spatial_merge_2x2(x_pre_merge, patches_h, patches_w)

        for _ in range(10):
            run_merge()
        torch.cuda.synchronize()
        results["spatial_merge"] = triton.testing.do_bench(run_merge, warmup=20, rep=100)

        # Print results
        print(f"\n{'Component':<25} {'Time (ms)':>10} {'Per Layer':>12}")
        print("-" * 50)
        print(f"{'patch_embed':<25} {results['patch_embed']:>10.3f} {'(1x)':>12}")
        print(f"{'qkv_proj':<25} {results['qkv_proj']:>10.3f} {f'({depth}x)':>12}")
        print(f"{'attention':<25} {results['attention']:>10.3f} {f'({depth}x)':>12}")
        print(f"{'o_proj':<25} {results['o_proj']:>10.3f} {f'({depth}x)':>12}")
        print(f"{'mlp_unfused':<25} {results['mlp_unfused']:>10.3f} {f'({depth}x)':>12}")
        print(f"{'mlp_fused':<25} {results['mlp_fused']:>10.3f} {f'({depth}x)':>12}")
        print(f"{'layer_norm':<25} {results['layer_norm']:>10.3f} {f'({depth*2}x)':>12}")
        print(f"{'layer_norm_fused':<25} {results['layer_norm_fused']:>10.3f} {f'({depth*2}x)':>12}")
        print(f"{'add_rms_unfused':<25} {results['add_rms_unfused']:>10.3f} {f'({depth*2}x)':>12}")
        print(f"{'add_rms_fused':<25} {results['add_rms_fused']:>10.3f} {f'({depth*2}x)':>12}")
        print(f"{'spatial_merge':<25} {results['spatial_merge']:>10.3f} {'(1x)':>12}")

        # Estimate full forward pass time
        print("\n--- Estimated Full Pass ---")
        full_unfused = (
            results["patch_embed"] +
            depth * (results["qkv_proj"] + results["attention"] + results["o_proj"] + results["mlp_unfused"]) +
            depth * 2 * results["layer_norm"] +
            results["spatial_merge"]
        )
        full_fused = (
            results["patch_embed"] +
            depth * (results["qkv_proj"] + results["attention"] + results["o_proj"] + results["mlp_fused"]) +
            depth * 2 * results["layer_norm_fused"] +
            results["spatial_merge"]
        )
        print(f"Unfused total: {full_unfused:.2f} ms")
        print(f"Fused total:   {full_fused:.2f} ms")
        print(f"Speedup:       {full_unfused/full_fused:.2f}x")

        # Breakdown by category
        attn_time = depth * (results["qkv_proj"] + results["attention"] + results["o_proj"])
        mlp_time = depth * results["mlp_fused"]
        norm_time = depth * 2 * results["layer_norm_fused"]
        other_time = results["patch_embed"] + results["spatial_merge"]

        total = attn_time + mlp_time + norm_time + other_time
        print(f"\n--- Time Breakdown (fused) ---")
        print(f"Attention:     {attn_time:6.2f} ms ({100*attn_time/total:5.1f}%)")
        print(f"MLP:           {mlp_time:6.2f} ms ({100*mlp_time/total:5.1f}%)")
        print(f"Norms:         {norm_time:6.2f} ms ({100*norm_time/total:5.1f}%)")
        print(f"Other:         {other_time:6.2f} ms ({100*other_time/total:5.1f}%)")

    print("\n" + "=" * 70)
    print("Profiling complete!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    with app.run():
        run_v11.remote()
