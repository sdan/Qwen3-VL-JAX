import modal
from pathlib import Path

app = modal.App("accel")
local_dir = Path(__file__).parent.resolve()

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "triton>=3.0.0")
    .add_local_dir(local_dir, remote_path="/root/kernels")
)

@app.function(image=base_image, gpu="H100", timeout=300)
def run_v2():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton
    from fused_ops import swiglu, fused_add_rms_norm, layer_norm

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print("=" * 70)
    print("Fused Ops Benchmark (Qwen3-VL Vision Encoder)")
    print("=" * 70)
    print(f"Device: {torch.cuda.get_device_name()}")
    print()

    # Qwen3-VL vision config
    hidden = 1152
    intermediate = 4304
    seq_lens = [256, 1024, 4096]

    # ========== SwiGLU ==========
    print("SwiGLU: silu(gate) * up")
    print("-" * 70)
    print(f"{'Seq':>6} | {'Fused ms':>10} | {'PyTorch ms':>11} | {'Speedup':>8} | {'MaxDiff':>10}")

    for seq in seq_lens:
        gate = torch.randn(seq, intermediate, device=device, dtype=dtype)
        up = torch.randn(seq, intermediate, device=device, dtype=dtype)

        # pytorch baseline
        def pytorch_swiglu():
            return torch.nn.functional.silu(gate) * up

        # fused
        def triton_swiglu():
            return swiglu(gate, up)

        # correctness
        ref = pytorch_swiglu()
        out = triton_swiglu()
        max_diff = (ref - out).abs().max().item()

        # warmup
        for _ in range(10):
            pytorch_swiglu()
            triton_swiglu()
        torch.cuda.synchronize()

        pt_ms = triton.testing.do_bench(pytorch_swiglu, warmup=20, rep=100)
        tr_ms = triton.testing.do_bench(triton_swiglu, warmup=20, rep=100)
        speedup = pt_ms / tr_ms

        print(f"{seq:>6} | {tr_ms:>10.3f} | {pt_ms:>11.3f} | {speedup:>7.2f}x | {max_diff:>10.6f}")

    print()

    # ========== Fused Add + RMSNorm ==========
    print("Fused Add + RMSNorm: rmsnorm(x + residual)")
    print("-" * 70)
    print(f"{'Seq':>6} | {'Fused ms':>10} | {'PyTorch ms':>11} | {'Speedup':>8} | {'MaxDiff':>10}")

    for seq in seq_lens:
        x = torch.randn(seq, hidden, device=device, dtype=dtype)
        residual = torch.randn(seq, hidden, device=device, dtype=dtype)
        weight = torch.ones(hidden, device=device, dtype=dtype)
        eps = 1e-6

        # pytorch baseline
        def pytorch_add_rms():
            s = x + residual
            var = s.float().pow(2).mean(-1, keepdim=True)
            return (s * torch.rsqrt(var + eps)) * weight, s

        # fused
        def triton_add_rms():
            return fused_add_rms_norm(x, residual, weight, eps)

        ref_out, ref_resid = pytorch_add_rms()
        out, resid_out = triton_add_rms()
        max_diff = max((ref_out - out).abs().max().item(), (ref_resid - resid_out).abs().max().item())

        for _ in range(10):
            pytorch_add_rms()
            triton_add_rms()
        torch.cuda.synchronize()

        pt_ms = triton.testing.do_bench(pytorch_add_rms, warmup=20, rep=100)
        tr_ms = triton.testing.do_bench(triton_add_rms, warmup=20, rep=100)
        speedup = pt_ms / tr_ms

        print(f"{seq:>6} | {tr_ms:>10.3f} | {pt_ms:>11.3f} | {speedup:>7.2f}x | {max_diff:>10.6f}")

    print()

    # ========== LayerNorm ==========
    print("LayerNorm (vision encoder)")
    print("-" * 70)
    print(f"{'Seq':>6} | {'Fused ms':>10} | {'PyTorch ms':>11} | {'Speedup':>8} | {'MaxDiff':>10}")

    for seq in seq_lens:
        x = torch.randn(seq, hidden, device=device, dtype=dtype)
        weight = torch.ones(hidden, device=device, dtype=dtype)
        bias = torch.zeros(hidden, device=device, dtype=dtype)
        eps = 1e-6

        # pytorch baseline
        ln = torch.nn.LayerNorm(hidden, eps=eps, device=device, dtype=dtype)
        ln.weight.data.fill_(1.0)
        ln.bias.data.fill_(0.0)

        def pytorch_ln():
            return ln(x)

        def triton_ln():
            return layer_norm(x, weight, bias, eps)

        ref = pytorch_ln()
        out = triton_ln()
        max_diff = (ref - out).abs().max().item()

        for _ in range(10):
            pytorch_ln()
            triton_ln()
        torch.cuda.synchronize()

        pt_ms = triton.testing.do_bench(pytorch_ln, warmup=20, rep=100)
        tr_ms = triton.testing.do_bench(triton_ln, warmup=20, rep=100)
        speedup = pt_ms / tr_ms

        print(f"{seq:>6} | {tr_ms:>10.3f} | {pt_ms:>11.3f} | {speedup:>7.2f}x | {max_diff:>10.6f}")

    print()
    print("=" * 70)
    print("Gradient Check")
    print("=" * 70)

    # quick gradient check
    x = torch.randn(64, hidden, device=device, dtype=torch.float32, requires_grad=True)
    w = torch.ones(hidden, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(hidden, device=device, dtype=torch.float32, requires_grad=True)

    out = layer_norm(x, w, b)
    loss = out.sum()
    loss.backward()
    print(f"LayerNorm grad check: x.grad exists={x.grad is not None}, w.grad exists={w.grad is not None}")

    gate = torch.randn(64, intermediate, device=device, dtype=torch.float32, requires_grad=True)
    up = torch.randn(64, intermediate, device=device, dtype=torch.float32, requires_grad=True)
    out = swiglu(gate, up)
    out.sum().backward()
    print(f"SwiGLU grad check: gate.grad exists={gate.grad is not None}, up.grad exists={up.grad is not None}")

    print()
    print("All benchmarks complete!")

    return True


if __name__ == "__main__":
    with app.run():
        run_v2.remote()
