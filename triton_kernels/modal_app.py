"""accel"""
import modal
from pathlib import Path

# App will deploy to "accel" on staging
app = modal.App("accel")

# Local kernels directory
local_dir = Path(__file__).parent

# Image with dependencies + local kernel code
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "triton>=3.0.0",
        "transformers",
        "safetensors",
        "numpy",
        "flask",
    )
    .add_local_dir(local_dir, remote_path="/root/kernels")
)


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v0():
    """Main application endpoint"""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import time

    print("=" * 60)
    print("Running workload")
    print("=" * 60)

    print(f"\nDevice: {torch.cuda.get_device_name()}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch: {torch.__version__}")

    from vision_attention import TritonVisionAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1024
    num_heads = 16
    head_dim = 64

    results = []

    for seq_len in [256, 1024, 4096, 16384]:
        print(f"\n--- Processing batch: {seq_len} ---")

        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)

        # Warmup
        for _ in range(10):
            _ = attn(x)
        torch.cuda.synchronize()

        # Run
        start = time.perf_counter()
        for _ in range(100):
            _ = attn(x)
        torch.cuda.synchronize()
        triton_ms = (time.perf_counter() - start) / 100 * 1000

        # Baseline
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(100):
            _ = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        torch.cuda.synchronize()
        baseline_ms = (time.perf_counter() - start) / 100 * 1000

        ratio = baseline_ms / triton_ms
        results.append((seq_len, triton_ms, baseline_ms, ratio))

        print(f"Custom:   {triton_ms:.3f} ms")
        print(f"Baseline: {baseline_ms:.3f} ms")
        print(f"Ratio:    {ratio:.2f}x")

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for seq_len, t_ms, b_ms, ratio in results:
        print(f"Batch {seq_len:<6}: {t_ms:.3f} ms vs {b_ms:.3f} ms ({ratio:.2f}x)")

    return results


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v3():
    """Kernel-level benchmark: Triton attention vs PyTorch SDPA (FlashAttention backend)."""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    from vision_attention import _fused_attention_kernel, next_power_of_2

    device = torch.device("cuda")
    dtype = torch.bfloat16

    batch = 1
    num_heads = 16
    head_dim = 64

    seq_lens = [256, 512, 1024, 2048, 4096, 8192, 16384]

    print("=" * 60)
    print("Benchmark: Triton _fused_attention_kernel vs torch SDPA")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(f"dtype={dtype} batch={batch} heads={num_heads} head_dim={head_dim}")

    results = []

    def sdpa_flash_only():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )

    for seq_len in seq_lens:
        q = torch.randn((batch, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        k = torch.randn((batch, num_heads, seq_len, head_dim), device=device, dtype=dtype)
        v = torch.randn((batch, num_heads, seq_len, head_dim), device=device, dtype=dtype)

        padded_head_dim = next_power_of_2(head_dim)

        def triton_attn():
            out = torch.empty_like(q)
            grid = lambda meta: (triton.cdiv(seq_len, meta["BLOCK_M"]), batch * num_heads)
            _fused_attention_kernel[grid](
                q, k, v, out,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                batch, num_heads, seq_len,
                HEAD_DIM=head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
            )
            return out

        def torch_sdpa():
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=False
            )

        # Compile + autotune once (excluded from timing).
        _ = triton_attn()
        with sdpa_flash_only():
            _ = torch_sdpa()
        torch.cuda.synchronize()

        # Quick correctness check (only once, to keep runtime bounded).
        if seq_len == seq_lens[0]:
            out_triton = triton_attn()
            with sdpa_flash_only():
                out_torch = torch_sdpa()
            torch.cuda.synchronize()
            diff = (out_triton - out_torch).float().abs()
            print(
                f"Correctness @ seq={seq_len}: max_abs={diff.max().item():.3e} "
                f"mean_abs={diff.mean().item():.3e}"
            )

        triton_ms = triton.testing.do_bench(triton_attn, warmup=50, rep=200)
        with sdpa_flash_only():
            baseline_ms = triton.testing.do_bench(torch_sdpa, warmup=50, rep=200)

        ratio = baseline_ms / triton_ms
        results.append(
            {
                "seq_len": seq_len,
                "triton_ms": float(triton_ms),
                "baseline_ms": float(baseline_ms),
                "ratio": float(ratio),
            }
        )
        print(
            f"seq={seq_len:>5}: triton={triton_ms:.3f}ms  "
            f"sdpa_flash={baseline_ms:.3f}ms  ratio={ratio:.2f}x"
        )

    return results


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v4():
    """Window-attention benchmark: fused RoPE vs unfused (RoPE + attention) vs SDPA Flash."""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    from vision_attention import _window_attention_kernel, _window_attention_rope_kernel, next_power_of_2

    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_heads = 16
    head_dim = 64
    half = head_dim // 2
    padded_head_dim = next_power_of_2(head_dim)
    padded_half_dim = next_power_of_2(half)

    total_tokens = 16384
    window_lens = [64, 128, 256, 512]

    print("=" * 60)
    print("Benchmark: window attention (fused RoPE) vs (RoPE + attention)")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(f"dtype={dtype} heads={num_heads} head_dim={head_dim} total_tokens={total_tokens}")

    results = []

    def sdpa_flash_only():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )

    for window_len in window_lens:
        num_windows = total_tokens // window_len
        total_seq = num_windows * window_len
        if total_seq == 0:
            continue

        cu_seqlens = torch.arange(
            0, total_seq + 1, window_len, device=device, dtype=torch.int32
        )
        max_ctx = window_len

        q = torch.randn((total_seq, num_heads, head_dim), device=device, dtype=dtype)
        k = torch.randn((total_seq, num_heads, head_dim), device=device, dtype=dtype)
        v = torch.randn((total_seq, num_heads, head_dim), device=device, dtype=dtype)

        # RoPE tables: [total_seq, head_dim//2]
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, half, 2, device=device).float() / half)
        )
        pos = torch.arange(total_seq, device=device, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [total_seq, half]
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)

        def triton_fused():
            out = torch.empty_like(q)
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

        def triton_unfused():
            cos_ = cos[:, None, :]
            sin_ = sin[:, None, :]

            q0, q1 = q[..., :half], q[..., half:]
            k0, k1 = k[..., :half], k[..., half:]
            q_rope = torch.cat([q0 * cos_ - q1 * sin_, q0 * sin_ + q1 * cos_], dim=-1)
            k_rope = torch.cat([k0 * cos_ - k1 * sin_, k0 * sin_ + k1 * cos_], dim=-1)

            out = torch.empty_like(q)
            grid = lambda meta: (triton.cdiv(max_ctx, meta["BLOCK_M"]), num_windows * num_heads)
            _window_attention_kernel[grid](
                q_rope, k_rope, v,
                out,
                cu_seqlens,
                q_rope.stride(0), q_rope.stride(1), q_rope.stride(2),
                k_rope.stride(0), k_rope.stride(1), k_rope.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                out.stride(0), out.stride(1), out.stride(2),
                num_windows, num_heads,
                max_ctx,
                HEAD_DIM=head_dim,
                PADDED_HEAD_DIM=padded_head_dim,
            )
            return out

        def sdpa_unfused():
            cos_ = cos[:, None, :]
            sin_ = sin[:, None, :]

            q0, q1 = q[..., :half], q[..., half:]
            k0, k1 = k[..., :half], k[..., half:]
            q_rope = torch.cat([q0 * cos_ - q1 * sin_, q0 * sin_ + q1 * cos_], dim=-1)
            k_rope = torch.cat([k0 * cos_ - k1 * sin_, k0 * sin_ + k1 * cos_], dim=-1)

            q_b = q_rope.reshape(num_windows, window_len, num_heads, head_dim).transpose(1, 2)
            k_b = k_rope.reshape(num_windows, window_len, num_heads, head_dim).transpose(1, 2)
            v_b = v.reshape(num_windows, window_len, num_heads, head_dim).transpose(1, 2)

            out = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_b, v_b, dropout_p=0.0, is_causal=False
            )
            return out.transpose(1, 2).reshape(total_seq, num_heads, head_dim)

        # Compile + autotune (excluded from timing).
        _ = triton_fused()
        _ = triton_unfused()
        with sdpa_flash_only():
            _ = sdpa_unfused()
        torch.cuda.synchronize()

        # Correctness check once (keeps runtime bounded).
        if window_len == window_lens[0]:
            out_fused = triton_fused()
            out_unfused = triton_unfused()
            with sdpa_flash_only():
                out_sdpa = sdpa_unfused()
            torch.cuda.synchronize()
            diff = (out_fused - out_unfused).float().abs()
            diff_sdpa = (out_fused - out_sdpa).float().abs()
            print(
                f"Correctness @ window={window_len}: max_abs={diff.max().item():.3e} "
                f"mean_abs={diff.mean().item():.3e}"
            )
            print(
                f"Correctness @ window={window_len} (vs SDPA): max_abs={diff_sdpa.max().item():.3e} "
                f"mean_abs={diff_sdpa.mean().item():.3e}"
            )

        fused_ms = triton.testing.do_bench(triton_fused, warmup=50, rep=200)
        unfused_ms = triton.testing.do_bench(triton_unfused, warmup=50, rep=200)
        with sdpa_flash_only():
            sdpa_ms = triton.testing.do_bench(sdpa_unfused, warmup=50, rep=200)
        speedup_unfused = unfused_ms / fused_ms
        speedup_sdpa = sdpa_ms / fused_ms

        results.append(
            {
                "window_len": window_len,
                "num_windows": num_windows,
                "fused_ms": float(fused_ms),
                "unfused_ms": float(unfused_ms),
                "sdpa_flash_ms": float(sdpa_ms),
                "speedup_unfused": float(speedup_unfused),
                "speedup_sdpa": float(speedup_sdpa),
            }
        )
        print(
            f"window={window_len:>4} (n={num_windows:>4}): "
            f"fused={fused_ms:.3f}ms unfused={unfused_ms:.3f}ms "
            f"sdpa_flash={sdpa_ms:.3f}ms speedup(unfused)={speedup_unfused:.2f}x "
            f"speedup(sdpa)={speedup_sdpa:.2f}x"
        )

    return results


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v5():
    """
    Benchmark with REAL Qwen3-VL vision encoder config.
    Tests non-power-of-2 head_dim (72) and compares against PyTorch SDPA.
    """
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    from vision_attention import (
        TritonVisionAttention,
        UnfusedVisionAttention,
        compute_vision_rope_2d,
        next_power_of_2,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Real Qwen3-VL vision encoder config (shared across all model sizes)
    hidden_size = 1152
    num_heads = 16
    head_dim = 72  # NOT a power of 2!
    window_size = 112

    print("=" * 70)
    print("Benchmark: Real Qwen3-VL Vision Config (head_dim=72)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(f"Config: hidden={hidden_size} heads={num_heads} head_dim={head_dim}")
    print(f"        padded_head_dim={next_power_of_2(head_dim)} padded_half={next_power_of_2(head_dim//2)}")
    print()

    # Test sequence lengths matching typical image sizes
    # 448x448 -> 32x32 patches -> 1024 tokens
    # 672x672 -> 48x48 patches -> 2304 tokens
    seq_lens = [256, 512, 1024, 2048, 4096]

    results = []

    def pick_hw(total_tokens: int, merge: int) -> tuple[int, int]:
        """Pick an (h, w) patch grid such that h*w == total_tokens and both divisible by merge."""
        root = int(total_tokens**0.5)
        for h in range(root, 0, -1):
            if total_tokens % h != 0:
                continue
            w = total_tokens // h
            if h % merge == 0 and w % merge == 0:
                return h, w
        raise ValueError(f"Could not find (h,w) for total_tokens={total_tokens} merge={merge}")

    for seq_len in seq_lens:
        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        h, w = pick_hw(seq_len, merge=2)
        grid_thw = [(1, h, w)]
        cos, sin = compute_vision_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=2,
            rope_theta=10000.0,
            device=device,
            dtype=dtype,
        )

        # Fused RoPE + Attention (our optimized kernel)
        fused = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)

        # Unfused (separate RoPE + attention)
        unfused = UnfusedVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)
        unfused.qkv.weight.data = fused.qkv.weight.data.clone()
        unfused.qkv.bias.data = fused.qkv.bias.data.clone()
        unfused.proj.weight.data = fused.proj.weight.data.clone()
        unfused.proj.bias.data = fused.proj.bias.data.clone()

        # Warmup + correctness check
        out_fused = fused(x, cos=cos, sin=sin)
        out_unfused = unfused(x, cos=cos, sin=sin)
        torch.cuda.synchronize()

        diff = (out_fused - out_unfused).float().abs()
        if seq_len == seq_lens[0]:
            print(f"Correctness: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")
            assert diff.max().item() < 1e-2, f"Output mismatch: {diff.max().item()}"
            print("Correctness check PASSED\n")

        # Benchmark
        fused_ms = triton.testing.do_bench(lambda: fused(x, cos=cos, sin=sin), warmup=50, rep=200)
        unfused_ms = triton.testing.do_bench(lambda: unfused(x, cos=cos, sin=sin), warmup=50, rep=200)

        speedup = unfused_ms / fused_ms

        results.append({
            "seq_len": seq_len,
            "fused_ms": float(fused_ms),
            "unfused_ms": float(unfused_ms),
            "speedup": float(speedup),
        })

        print(f"seq={seq_len:>5}: fused={fused_ms:.3f}ms unfused={unfused_ms:.3f}ms speedup={speedup:.2f}x")

    print()
    print("=" * 70)
    print("Summary: Fused RoPE+Attention speedup over Unfused")
    print("=" * 70)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"Average speedup: {avg_speedup:.2f}x")

    return results


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v6():
    """End-to-end benchmark: fused vision blocks vs unfused vs SDPA Flash."""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import torch.nn as nn
    import triton

    from vision_attention import (
        TritonVisionAttention,
        UnfusedVisionAttention,
        compute_vision_rope_2d,
    )
    from window_attn import get_window_index, window_shuffle

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Qwen3-VL-ish defaults (tweak here as needed).
    hidden_size = 1024
    num_heads = 16
    head_dim = hidden_size // num_heads
    intermediate_size = 4096
    depth = 24
    patch_size = 14
    spatial_merge_size = 2
    window_size = 112
    # Blocks listed here use full attention; all others use window attention.
    fullatt_block_indexes = tuple(range(0, depth, 4))
    fullatt_set = set(int(i) for i in fullatt_block_indexes)

    grid_thw = [(1, 128, 128)]  # 128*128 == 16384 tokens
    seq_len = int(sum(int(t) * int(h) * int(w) for (t, h, w) in grid_thw))
    unit = spatial_merge_size ** 2
    assert seq_len % unit == 0, "seq_len must be divisible by merge unit"

    print("=" * 70)
    print("Benchmark: Vision encoder (fused vs unfused vs SDPA Flash)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}  Triton: {triton.__version__}")
    print(
        f"dtype={dtype} hidden={hidden_size} heads={num_heads} head_dim={head_dim} "
        f"mlp={intermediate_size} depth={depth} seq={seq_len} "
        f"window_size={window_size} patch={patch_size} merge={spatial_merge_size} "
        f"fullatt={sorted(fullatt_set)}"
    )

    def sdpa_flash_only():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            )

    class VisionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            return self.fc2(x)

    class VisionBlock(nn.Module):
        def __init__(self, attention_cls):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.attn = attention_cls(hidden_size, num_heads, head_dim)
            self.mlp = VisionMLP()

        def forward(self, x, cu_seqlens, cos, sin):
            x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            x = x + self.mlp(self.norm2(x))
            return x

    class PatchMerger(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.fc1 = nn.Linear(hidden_size * unit, hidden_size * unit, bias=True, dtype=dtype)
            self.fc2 = nn.Linear(hidden_size * unit, hidden_size, bias=True, dtype=dtype)

        def forward(self, x):
            x = self.norm(x)
            x = x.reshape(-1, hidden_size * unit)
            x = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
            return self.fc2(x)

    class SdpaVisionAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True, dtype=dtype)
            self.proj = nn.Linear(hidden_size, hidden_size, bias=True, dtype=dtype)

        def forward(self, x, cu_seqlens=None, cos=None, sin=None):
            seq = x.shape[0]
            qkv = self.qkv(x).reshape(seq, 3, num_heads, head_dim)
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

            if cos is None or sin is None:
                raise ValueError("cos/sin must be provided for SDPA benchmark")

            half = head_dim // 2
            cos_ = cos[:, None, :].to(dtype=q.dtype)
            sin_ = sin[:, None, :].to(dtype=q.dtype)
            q0, q1 = q[..., :half], q[..., half:]
            k0, k1 = k[..., :half], k[..., half:]
            q = torch.cat([q0 * cos_ - q1 * sin_, q0 * sin_ + q1 * cos_], dim=-1)
            k = torch.cat([k0 * cos_ - k1 * sin_, k0 * sin_ + k1 * cos_], dim=-1)

            if cu_seqlens is None:
                num_windows = 1
                window = seq
            else:
                num_windows = cu_seqlens.shape[0] - 1
                window = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

            if num_windows * window != seq:
                raise ValueError("SDPA benchmark assumes equal-length windows")

            q_b = q.reshape(num_windows, window, num_heads, head_dim).transpose(1, 2)
            k_b = k.reshape(num_windows, window, num_heads, head_dim).transpose(1, 2)
            v_b = v.reshape(num_windows, window, num_heads, head_dim).transpose(1, 2)

            out = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_b, v_b, dropout_p=0.0, is_causal=False
            )
            out = out.transpose(1, 2).reshape(seq, hidden_size)
            return self.proj(out)

    class VisionEncoder(nn.Module):
        def __init__(self, attention_cls):
            super().__init__()
            self.blocks = nn.ModuleList([VisionBlock(attention_cls) for _ in range(depth)])
            self.merger = PatchMerger()

        def forward(self, x, cu_full, cu_window, cos, sin, *, window_idx=None):
            use_window = (
                cu_window is not None
                and window_idx is not None
                and len(fullatt_set) < len(self.blocks)
            )
            for i, blk in enumerate(self.blocks):
                cu = cu_full if (not use_window or i in fullatt_set) else cu_window
                x = blk(x, cu, cos, sin)
            x = self.merger(x)
            if use_window:
                reverse_idx = torch.argsort(window_idx)
                x = x[reverse_idx, :]
            return x

    # Qwen3-VL window shuffle + packed cu_seqlens for window attention.
    use_window = len(fullatt_set) < depth
    window_idx = None
    cu_window = None
    if use_window:
        window_idx, cu_window = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        assert cu_window[0].item() == 0 and cu_window[-1].item() == seq_len

    # Full-attention cu_seqlens (per image/video item).
    token_counts = [int(t) * int(h) * int(w) for (t, h, w) in grid_thw]
    cu_full = torch.tensor(
        [0] + list(torch.tensor(token_counts, device=device).cumsum(0).tolist()),
        device=device,
        dtype=torch.int32,
    )
    assert cu_full[-1].item() == seq_len

    # RoPE tables: Qwen3-VL-style 2D vision RoPE (h, w), optionally reordered to match window shuffle.
    cos, sin = compute_vision_rope_2d(
        grid_thw,
        head_dim=head_dim,
        spatial_merge_size=spatial_merge_size,
        rope_theta=10000.0,
        device=device,
        dtype=dtype,
        window_idx=window_idx,
    )

    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
    if use_window:
        x = window_shuffle(x, window_idx, spatial_merge_size=spatial_merge_size)

    enc_fused = VisionEncoder(TritonVisionAttention).to(device, dtype)
    enc_unfused = VisionEncoder(UnfusedVisionAttention).to(device, dtype)
    enc_sdpa = VisionEncoder(lambda *_args, **_kwargs: SdpaVisionAttention()).to(device, dtype)

    enc_unfused.load_state_dict(enc_fused.state_dict(), strict=True)
    enc_sdpa.load_state_dict(enc_fused.state_dict(), strict=True)

    # Compile/autotune once (excluded from timing).
    _ = enc_fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    _ = enc_unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    with sdpa_flash_only():
        _ = enc_sdpa(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    torch.cuda.synchronize()

    out_fused = enc_fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    out_unfused = enc_unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    with sdpa_flash_only():
        out_sdpa = enc_sdpa(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
    torch.cuda.synchronize()

    diff_unfused = (out_fused - out_unfused).float().abs()
    diff_sdpa = (out_fused - out_sdpa).float().abs()
    print(
        f"Correctness: fused vs unfused max_abs={diff_unfused.max().item():.3e} "
        f"mean_abs={diff_unfused.mean().item():.3e}"
    )
    print(
        f"Correctness: fused vs sdpa max_abs={diff_sdpa.max().item():.3e} "
        f"mean_abs={diff_sdpa.mean().item():.3e}"
    )

    fused_ms = triton.testing.do_bench(
        lambda: enc_fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx),
        warmup=10,
        rep=30,
    )
    unfused_ms = triton.testing.do_bench(
        lambda: enc_unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx),
        warmup=10,
        rep=30,
    )
    with sdpa_flash_only():
        sdpa_ms = triton.testing.do_bench(
            lambda: enc_sdpa(x, cu_full, cu_window, cos, sin, window_idx=window_idx),
            warmup=10,
            rep=30,
        )

    print("-" * 70)
    print(f"Fused (Triton RoPE+attn): {fused_ms:.3f} ms")
    print(f"Unfused (RoPE + Triton): {unfused_ms:.3f} ms  speedup={unfused_ms / fused_ms:.2f}x")
    print(f"SDPA Flash (RoPE + SDPA): {sdpa_ms:.3f} ms  speedup={sdpa_ms / fused_ms:.2f}x")

    return {
        "seq_len": seq_len,
        "grid_thw": grid_thw,
        "window_size": window_size,
        "patch_size": patch_size,
        "spatial_merge_size": spatial_merge_size,
        "num_windows": int(cu_window.shape[0] - 1) if cu_window is not None else 1,
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "depth": depth,
        "fullatt_block_indexes": list(sorted(fullatt_set)),
        "fused_ms": float(fused_ms),
        "unfused_ms": float(unfused_ms),
        "sdpa_ms": float(sdpa_ms),
        "speedup_unfused": float(unfused_ms / fused_ms),
        "speedup_sdpa": float(sdpa_ms / fused_ms),
    }


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v7():
    """Benchmark 2x2 spatial merge: Triton kernel vs PyTorch reshape/permute."""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton

    from spatial_merge import spatial_merge_2x2

    device = torch.device("cuda")
    dtype = torch.bfloat16

    h = 32
    w = 32
    c = 1024

    x = torch.randn((h * w, c), device=device, dtype=dtype)

    def triton_merge():
        return spatial_merge_2x2(x, h, w)

    def torch_merge():
        x_hw = x.reshape(h // 2, 2, w // 2, 2, c)
        x_hw = x_hw.permute(0, 2, 1, 3, 4)
        return x_hw.reshape(-1, 4 * c)

    _ = triton_merge()
    _ = torch_merge()
    torch.cuda.synchronize()

    out_triton = triton_merge()
    out_torch = torch_merge()
    torch.cuda.synchronize()

    diff = (out_triton - out_torch).float().abs()
    print(
        f"Correctness: max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}"
    )

    triton_ms = triton.testing.do_bench(triton_merge, warmup=50, rep=200)
    torch_ms = triton.testing.do_bench(torch_merge, warmup=50, rep=200)

    print("=" * 60)
    print("Benchmark: spatial merge 2x2")
    print("=" * 60)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"dtype={dtype} H={h} W={w} C={c}")
    print(f"Triton: {triton_ms:.3f} ms")
    print(f"PyTorch: {torch_ms:.3f} ms")
    print(f"Speedup: {torch_ms / triton_ms:.2f}x")

    return {
        "H": h,
        "W": w,
        "C": c,
        "triton_ms": float(triton_ms),
        "torch_ms": float(torch_ms),
        "speedup": float(torch_ms / triton_ms),
    }


@app.function(
    image=base_image,
    gpu="H100",
    timeout=300,
)
def run_v1():
    """Verify system health - tests both power-of-2 and non-power-of-2 head_dim"""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch

    print("Running health check...")

    from vision_attention import TritonVisionAttention, next_power_of_2

    device = torch.device("cuda")
    dtype = torch.float32

    # Test 1: Standard power-of-2 head_dim
    print("\n--- Test 1: head_dim=64 (power of 2) ---")
    hidden_size = 1024
    num_heads = 16
    head_dim = 64
    seq_len = 256

    attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    out = attn(x)

    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean().item():.6f}")
    print(f"Output std: {out.std().item():.6f}")

    assert not torch.isnan(out).any(), "NaN detected"
    assert not torch.isinf(out).any(), "Inf detected"
    print("PASSED")

    # Test 2: Qwen3-VL's non-power-of-2 head_dim
    print("\n--- Test 2: head_dim=72 (Qwen3-VL config) ---")
    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    print(f"Padded: head_dim={head_dim} -> {next_power_of_2(head_dim)}, half={head_dim//2} -> {next_power_of_2(head_dim//2)}")

    attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    out = attn(x)

    print(f"Output shape: {out.shape}")
    print(f"Output mean: {out.mean().item():.6f}")
    print(f"Output std: {out.std().item():.6f}")

    assert not torch.isnan(out).any(), "NaN detected"
    assert not torch.isinf(out).any(), "Inf detected"
    print("PASSED")

    print("\n" + "=" * 40)
    print("All health checks PASSED")
    print("=" * 40)
    return True


@app.function(
    image=base_image,
    gpu="H100",
    timeout=300,
)
def run_v2():
    """Collect performance metrics"""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    from torch.profiler import profile, ProfilerActivity

    print("Collecting metrics...")

    from vision_attention import TritonVisionAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1024
    num_heads = 16
    seq_len = 4096

    attn = TritonVisionAttention(hidden_size, num_heads).to(device, dtype)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    for _ in range(10):
        _ = attn(x)
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for _ in range(10):
            _ = attn(x)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
    print(f"\nPeak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return True


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v8():
    """Full data processing pipeline"""
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch

    print("=" * 60)
    print("Data Pipeline")
    print("=" * 60)

    # Future: full vision encoder
    print("Pipeline ready")

    return True


@app.function(
    image=base_image,
    gpu="H100",
    timeout=600,
)
def run_v9():
    """
    Profile a simulated 32-layer Qwen3-VL vision encoder.
    Breaks down time by operation type to identify optimization targets.
    """
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import torch.nn as nn
    import triton
    from torch.profiler import profile, ProfilerActivity

    from vision_attention import (
        _fused_attention_kernel,
        _window_attention_rope_kernel,
        next_power_of_2,
    )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Real Qwen3-VL vision encoder config (shared across 2B/8B/32B/MoE)
    hidden_size = 1152
    num_heads = 16
    head_dim = 72  # NOT hidden_size // num_heads! Qwen uses explicit 72
    intermediate_size = 4608  # 4x hidden
    num_layers = 27  # Real depth from config
    window_size = 112  # typical for 448x448 image with patch=14
    spatial_merge = 2

    # Simulate 448x448 image -> 32x32 patches -> 1024 tokens
    # After merge: 256 tokens
    seq_len = 1024
    num_windows = seq_len // window_size

    print("=" * 70)
    print("Profile: Real Qwen3-VL Vision Encoder (27-layer, head_dim=72)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Config: hidden={hidden_size} heads={num_heads} head_dim={head_dim}")
    print(f"        layers={num_layers} seq={seq_len} window={window_size}")
    print(f"        intermediate={intermediate_size} dtype={dtype}")
    print()

    # Create components
    class LayerNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm = nn.LayerNorm(dim, dtype=dtype)
        def forward(self, x):
            return self.norm(x)

    class MLP(nn.Module):
        def __init__(self, hidden, intermediate):
            super().__init__()
            self.fc1 = nn.Linear(hidden, intermediate, dtype=dtype)
            self.fc2 = nn.Linear(intermediate, hidden, dtype=dtype)
        def forward(self, x):
            return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

    class QKVProj(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.qkv = nn.Linear(hidden, 3 * hidden, dtype=dtype)
        def forward(self, x):
            return self.qkv(x)

    class OutProj(nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.proj = nn.Linear(hidden, hidden, dtype=dtype)
        def forward(self, x):
            return self.proj(x)

    # Build layers
    norms1 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(num_layers)]).to(device)
    norms2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(num_layers)]).to(device)
    qkv_projs = nn.ModuleList([QKVProj(hidden_size) for _ in range(num_layers)]).to(device)
    out_projs = nn.ModuleList([OutProj(hidden_size) for _ in range(num_layers)]).to(device)
    mlps = nn.ModuleList([MLP(hidden_size, intermediate_size) for _ in range(num_layers)]).to(device)

    # cu_seqlens for window attention
    cu_seqlens = torch.arange(0, num_windows + 1, device=device, dtype=torch.int32) * window_size

    # RoPE tables
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim // 2, device=device).float() / (head_dim // 2)))
    pos = torch.arange(window_size, device=device).float()
    emb = torch.outer(pos, inv_freq)
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    # Input
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    # Timing storage
    times = {
        'layernorm': 0.0,
        'qkv_proj': 0.0,
        'attention': 0.0,
        'out_proj': 0.0,
        'mlp': 0.0,
        'total': 0.0,
    }

    def time_op(fn, name, reps=50):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        # Warmup
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        start.record()
        for _ in range(reps):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / reps

    # Profile individual ops
    print("Profiling individual operations (averaged over 50 runs)...")
    print("-" * 70)

    # LayerNorm
    ln_time = time_op(lambda: norms1[0](x), 'layernorm')
    times['layernorm'] = ln_time * num_layers * 2  # 2 per layer
    print(f"LayerNorm:     {ln_time:.4f} ms/op × {num_layers*2} = {times['layernorm']:.3f} ms")

    # QKV projection
    qkv_time = time_op(lambda: qkv_projs[0](x), 'qkv_proj')
    times['qkv_proj'] = qkv_time * num_layers
    print(f"QKV Proj:      {qkv_time:.4f} ms/op × {num_layers} = {times['qkv_proj']:.3f} ms")

    # Attention (our Triton kernel)
    qkv = qkv_projs[0](x)  # [seq, 3*hidden]
    qkv_reshaped = qkv.reshape(seq_len, 3, num_heads, head_dim)
    q, k, v = qkv_reshaped[:, 0], qkv_reshaped[:, 1], qkv_reshaped[:, 2]

    # Compute padded dimensions for non-power-of-2 head_dim
    padded_head_dim = next_power_of_2(head_dim)
    padded_half_dim = next_power_of_2(head_dim // 2)

    def run_attention():
        out = torch.empty_like(q)
        grid = lambda meta: (triton.cdiv(window_size, meta["BLOCK_M"]), num_windows * num_heads)
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
            window_size,
            HEAD_DIM=head_dim,
            PADDED_HEAD_DIM=padded_head_dim,
            PADDED_HALF_DIM=padded_half_dim,
        )
        return out

    attn_time = time_op(run_attention, 'attention')
    times['attention'] = attn_time * num_layers
    print(f"Attention:     {attn_time:.4f} ms/op × {num_layers} = {times['attention']:.3f} ms")

    # Out projection
    attn_out = run_attention().reshape(seq_len, hidden_size)
    out_time = time_op(lambda: out_projs[0](attn_out), 'out_proj')
    times['out_proj'] = out_time * num_layers
    print(f"Out Proj:      {out_time:.4f} ms/op × {num_layers} = {times['out_proj']:.3f} ms")

    # MLP
    mlp_time = time_op(lambda: mlps[0](x), 'mlp')
    times['mlp'] = mlp_time * num_layers
    print(f"MLP:           {mlp_time:.4f} ms/op × {num_layers} = {times['mlp']:.3f} ms")

    # Total
    times['total'] = sum(v for k, v in times.items() if k != 'total')

    print("-" * 70)
    print(f"{'TOTAL:':<14} {times['total']:.3f} ms for {num_layers} layers")
    print()

    # Breakdown
    print("=" * 70)
    print("Time Breakdown (% of total)")
    print("=" * 70)
    for name, t in sorted(times.items(), key=lambda x: -x[1]):
        if name != 'total':
            pct = t / times['total'] * 100
            bar = '█' * int(pct / 2)
            print(f"{name:<12}: {t:>7.3f} ms ({pct:>5.1f}%) {bar}")

    print()
    print("=" * 70)
    print("Optimization Recommendations")
    print("=" * 70)

    # Sort by time
    sorted_ops = sorted([(k, v) for k, v in times.items() if k != 'total'], key=lambda x: -x[1])
    top_op = sorted_ops[0][0]

    if top_op == 'mlp':
        print("→ MLP dominates. Consider:")
        print("  1. Fused GELU (already using F.gelu which should fuse)")
        print("  2. Fuse LayerNorm2 + MLP into single kernel")
        print("  3. Use FP8 for MLP (H100 native support)")
    elif top_op == 'attention':
        print("→ Attention dominates. Consider:")
        print("  1. Already using fused RoPE+attention ✓")
        print("  2. Try larger block sizes for this head_dim")
        print("  3. Profile TensorCore utilization")
    elif top_op == 'qkv_proj' or top_op == 'out_proj':
        print("→ Linear projections dominate. Consider:")
        print("  1. Hard to beat cuBLAS for standalone GEMMs")
        print("  2. Fuse QKV into attention kernel (advanced)")
        print("  3. Use FP8 quantization")
    elif top_op == 'layernorm':
        print("→ LayerNorm dominates. Consider:")
        print("  1. Fuse LayerNorm into attention/MLP kernels")
        print("  2. Use RMSNorm instead (slightly faster)")

    # Memory stats
    print()
    print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    return times


@app.local_entrypoint()
def main():
    """Entry point"""
    print("Starting...")
    run_v1.remote()
    results = run_v0.remote()
    run_v2.remote()
    print("\nComplete.")
