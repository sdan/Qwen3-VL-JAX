# Complete Instructions: Qwen3-VL Triton Kernels

## Objective
Build custom Triton GPU kernels for Qwen3-VL's vision encoder that match or beat PyTorch's cuDNN FlashAttention.

## Current State
- ✅ Basic Triton attention kernel exists (`vision_attention.py`)
- ✅ Modal deployment works (`modal_app.py` on staging/accel)
- ❌ Kernel is 1.5-33x slower than baseline
- ❌ Block sizes not optimized for H100
- ❌ Missing autotuning

## Environment
```bash
# Run on Modal (H100)
cd /Users/sdan/Developer/Qwen3-VL-JAX/triton_kernels
modal run --env staging modal_app.py::run_v1
modal run --env staging modal_app.py::run_v0  # Full benchmark
```

No ncu/Nsight needed - use `torch.profiler` and `triton.testing.do_bench`.

---

## TASK 1: Fix Block Sizes and Add Autotuning

### File: `vision_attention.py`

Replace the `_fused_attention_kernel` with autotuned version:

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
    ],
    key=['N_CTX', 'HEAD_DIM'],
)
@triton.jit
def _fused_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ... kernel code
```

### Key Changes Needed:

1. **Add HEAD_DIM as constexpr parameter** (remove BLOCK_DMODEL, use HEAD_DIM)

2. **Use exp2 instead of exp** (faster on GPU):
```python
# Before
p = tl.exp(qk - m_i_new[:, None])

# After
LOG2E = 1.44269504089  # log2(e)
qk_scale = LOG2E / tl.sqrt(tl.cast(HEAD_DIM, tl.float32))
# ... in loop:
p = tl.math.exp2(qk - m_i_new[:, None])
```

3. **Keep dtypes consistent** - no conversions in hot path:
```python
# Load in original dtype, compute attention in original dtype
q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
# Scale without dtype conversion
q = q * qk_scale  # qk_scale should be same dtype
```

4. **Fix grid function**:
```python
grid = lambda meta: (
    triton.cdiv(N_CTX, meta['BLOCK_M']),  # Number of M blocks
    Z * H,  # batch * heads
)
```

---

## TASK 2: Add Proper Benchmarking

### File: `modal_app.py`

Add `triton.testing.do_bench` for accurate timing:

```python
@app.function(image=base_image, gpu="H100", timeout=600)
def benchmark():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    import triton
    from vision_attention import TritonVisionAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1024
    num_heads = 16
    head_dim = 64

    results = []

    for seq_len in [256, 512, 1024, 2048, 4096, 8192, 16384]:
        x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)
        attn = TritonVisionAttention(hidden_size, num_heads, head_dim).to(device, dtype)

        # Triton benchmark (handles warmup automatically)
        triton_ms = triton.testing.do_bench(lambda: attn(x), warmup=100, rep=500)

        # Baseline
        q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        baseline_ms = triton.testing.do_bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v),
            warmup=100, rep=500
        )

        ratio = baseline_ms / triton_ms
        results.append({
            'seq_len': seq_len,
            'triton_ms': triton_ms,
            'baseline_ms': baseline_ms,
            'ratio': ratio,
            'status': '✓' if ratio >= 0.9 else '✗'
        })

        print(f"seq={seq_len:>5}: triton={triton_ms:.3f}ms baseline={baseline_ms:.3f}ms ratio={ratio:.2f}x {results[-1]['status']}")

    return results
```

---

## TASK 3: Profile Memory Bandwidth

Add this function to understand if we're memory or compute bound:

```python
@app.function(image=base_image, gpu="H100", timeout=300)
def profile_bandwidth():
    import sys
    sys.path.insert(0, "/root/kernels")

    import torch
    from torch.profiler import profile, ProfilerActivity, schedule
    from vision_attention import TritonVisionAttention

    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = 4096
    hidden_size = 1024
    num_heads = 16

    attn = TritonVisionAttention(hidden_size, num_heads).to(device, dtype)
    x = torch.randn(seq_len, hidden_size, device=device, dtype=dtype)

    # Warmup
    for _ in range(50):
        _ = attn(x)
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for _ in range(10):
            _ = attn(x)
        torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Memory bandwidth calculation
    # Q, K, V each: seq_len * num_heads * head_dim * 2 bytes (bf16)
    # Output: same size
    # Total: 4 * seq_len * hidden_size * 2 bytes
    bytes_accessed = 4 * seq_len * hidden_size * 2
    cuda_time_s = sum(e.cuda_time for e in prof.key_averages()) / 1e6
    bandwidth_gb_s = (bytes_accessed * 10) / cuda_time_s / 1e9

    print(f"\nEstimated bandwidth: {bandwidth_gb_s:.1f} GB/s")
    print(f"H100 theoretical: 3350 GB/s")
    print(f"Utilization: {bandwidth_gb_s/3350*100:.1f}%")

    return True
```

---

## TASK 4: Implement Fused QKV + RoPE Kernel

This is where we WIN - fuse operations to reduce memory traffic.

### New file: `fused_qkv_rope.py`

```python
import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_qkv_rope_kernel(
    X,           # Input [seq, hidden]
    W_qkv,       # QKV weights [hidden, 3*hidden]
    B_qkv,       # QKV bias [3*hidden]
    cos, sin,    # RoPE tables [seq, head_dim/2]
    Q_out, K_out, V_out,  # Outputs [seq, heads, head_dim]
    seq_len, hidden_size, num_heads, head_dim,
    stride_x_seq, stride_x_hidden,
    stride_w_in, stride_w_out,
    stride_o_seq, stride_o_head, stride_o_dim,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    """
    Fused kernel that does:
    1. QKV = X @ W_qkv + B_qkv
    2. Split into Q, K, V
    3. Apply RoPE to Q and K
    4. Store Q, K, V

    All in one kernel = 1 read of X, 1 write of Q,K,V
    vs 3+ separate kernel launches
    """
    pid_seq = tl.program_id(0)
    pid_head = tl.program_id(1)

    # This position in sequence
    seq_idx = pid_seq * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    mask_seq = seq_idx < seq_len

    # Load X for this sequence position
    # ... (implement matmul tile)

    # Apply RoPE
    half_dim = head_dim // 2
    cos_val = tl.load(cos + seq_idx[:, None] * half_dim + tl.arange(0, half_dim)[None, :])
    sin_val = tl.load(sin + seq_idx[:, None] * half_dim + tl.arange(0, half_dim)[None, :])

    # RoPE: (x1, x2) -> (x1*cos - x2*sin, x1*sin + x2*cos)
    q1, q2 = q[:, :half_dim], q[:, half_dim:]
    q_rope = tl.cat([q1 * cos_val - q2 * sin_val, q1 * sin_val + q2 * cos_val], dim=1)

    # Same for K
    k1, k2 = k[:, :half_dim], k[:, half_dim:]
    k_rope = tl.cat([k1 * cos_val - k2 * sin_val, k1 * sin_val + k2 * cos_val], dim=1)

    # Store
    tl.store(Q_out + ..., q_rope, mask=mask_seq[:, None])
    tl.store(K_out + ..., k_rope, mask=mask_seq[:, None])
    tl.store(V_out + ..., v, mask=mask_seq[:, None])  # V doesn't get RoPE
```

---

## TASK 5: Window Attention for Qwen3-VL

Qwen3-VL uses window attention in vision encoder. Each window is independent.

### Update `_window_attention_kernel`:

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4),
    ],
    key=['HEAD_DIM'],
)
@triton.jit
def _window_attention_kernel(
    Q, K, V, Out,
    cu_seqlens,  # [num_windows + 1] - cumulative lengths
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    num_windows, num_heads,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    window_id = tl.program_id(0)
    head_id = tl.program_id(1)

    # Get window boundaries
    start = tl.load(cu_seqlens + window_id)
    end = tl.load(cu_seqlens + window_id + 1)
    window_len = end - start

    # Standard FlashAttention within this window
    # ... (same as fused attention but bounded by window)
```

---

## TASK 6: Spatial Merge Kernel

Qwen3-VL merges 2x2 patches. Simple fused kernel:

```python
@triton.jit
def _spatial_merge_kernel(
    X,      # Input [H*W, C]
    Out,    # Output [H*W/4, C*4] or [H*W/4, C] with projection
    H, W, C,
    BLOCK_C: tl.constexpr,
):
    """
    Merge 2x2 spatial patches:

    Before: [0,0] [0,1]    After: [concat or avg]
            [1,0] [1,1]

    This reduces sequence length by 4x.
    """
    pid = tl.program_id(0)  # Which output position

    # Calculate input positions for 2x2 patch
    out_h = pid // (W // 2)
    out_w = pid % (W // 2)

    in_positions = [
        (out_h * 2 + 0) * W + (out_w * 2 + 0),
        (out_h * 2 + 0) * W + (out_w * 2 + 1),
        (out_h * 2 + 1) * W + (out_w * 2 + 0),
        (out_h * 2 + 1) * W + (out_w * 2 + 1),
    ]

    # Load all 4 patches
    offs_c = tl.arange(0, BLOCK_C)
    x00 = tl.load(X + in_positions[0] * C + offs_c)
    x01 = tl.load(X + in_positions[1] * C + offs_c)
    x10 = tl.load(X + in_positions[2] * C + offs_c)
    x11 = tl.load(X + in_positions[3] * C + offs_c)

    # Concatenate (or average)
    out = tl.cat([x00, x01, x10, x11], dim=0)  # [4*C]

    tl.store(Out + pid * (C * 4) + tl.arange(0, BLOCK_C * 4), out)
```

---

## Success Criteria

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Attention speed | ≥0.9x baseline | `modal run --env staging modal_app.py::run_v3` |
| Fused QKV+RoPE | ≥1.2x vs separate | Compare fused vs unfused |
| Memory usage | ≤ baseline | `torch.cuda.max_memory_allocated()` |
| Full encoder | ≥1.1x baseline | End-to-end benchmark |

---

## Commands Reference

```bash
# Test health
modal run --env staging modal_app.py::run_v1

# Run benchmark
modal run --env staging modal_app.py::run_v0

# Run profiler
modal run --env staging modal_app.py::run_v2

# Deploy persistently
modal deploy --env staging modal_app.py
```

---

## Files to Modify

1. `vision_attention.py` - Add autotuning, fix block sizes, exp2
2. `modal_app.py` - Add better benchmarking with do_bench
3. NEW: `fused_qkv_rope.py` - Fused QKV projection + RoPE
4. NEW: `spatial_merge.py` - 2x2 patch merging

---

## Order of Operations

1. **First**: Fix autotuning in `_fused_attention_kernel` → Match baseline
2. **Second**: Add fused QKV+RoPE kernel → Beat baseline
3. **Third**: Add window attention support → Qwen3-VL compatible
4. **Fourth**: Add spatial merge → Complete vision encoder
5. **Fifth**: Benchmark full encoder end-to-end → Prove value

---

## No External Dependencies Needed

Everything is available:
- ✅ Triton (on Modal)
- ✅ PyTorch (on Modal)
- ✅ H100 access (Modal staging/accel)
- ✅ Profiling (torch.profiler, triton.testing.do_bench)
- ✅ Reference code (saved in `reference/triton_flash_attn.py`)
