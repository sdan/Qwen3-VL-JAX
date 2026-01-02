# Qwen3-VL Triton Kernels: Technical Plan

## What We're Building

Custom GPU kernels for **Qwen3-VL's vision encoder** - the part that processes images before the language model sees them. This is novel because no public Triton implementation exists for Qwen3-VL's specific architecture.

## End Goal

Beat or match PyTorch's `scaled_dot_product_attention` (cuDNN FlashAttention) with a custom Triton kernel that:
1. Fuses multiple operations (QKV projection + RoPE + attention)
2. Supports Qwen3-VL's window attention pattern
3. Supports 3D mRoPE (temporal, height, width position encoding for video)

**Success metric**: Match baseline at seq_len=4096+, then fuse more ops to win overall.

---

## Core Technology Explained

### 1. FlashAttention (What We're Implementing)

Standard attention is O(N²) memory because you materialize the full NxN attention matrix:
```
Attention(Q,K,V) = softmax(QK^T / sqrt(d)) @ V
```

**FlashAttention** tiles this computation:
- Load small blocks of Q, K, V into SRAM (fast on-chip memory)
- Compute partial attention for each block
- Use "online softmax" to combine blocks without storing full matrix
- Result: O(N) memory, same O(N²) compute but much faster due to memory hierarchy

```
┌─────────────────────────────────────────────────┐
│  HBM (80GB, slow)                               │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐            │
│  │ Q full  │ │ K full  │ │ V full  │            │
│  └────┬────┘ └────┬────┘ └────┬────┘            │
│       │           │           │                 │
│       ▼           ▼           ▼                 │
│  ┌─────────────────────────────────┐            │
│  │  SRAM (20MB per SM, fast)       │            │
│  │  ┌──────┐ ┌──────┐ ┌──────┐     │            │
│  │  │Q tile│ │K tile│ │V tile│     │  ◄── Work here!
│  │  └──────┘ └──────┘ └──────┘     │            │
│  │       │                         │            │
│  │       ▼                         │            │
│  │  Compute QK^T, softmax, @V      │            │
│  │  (all in SRAM, no HBM traffic)  │            │
│  └─────────────────────────────────┘            │
└─────────────────────────────────────────────────┘
```

### 2. Online Softmax (The Key Trick)

Normal softmax needs the full row to compute max and sum. Online softmax updates incrementally:

```python
# For each new block of attention scores:
m_new = max(m_old, max(new_block))           # Running max
l_new = l_old * exp(m_old - m_new) + sum(exp(new_block - m_new))  # Running sum
acc = acc * exp(m_old - m_new) + softmax(new_block) @ V_block     # Running output
```

This lets us process K,V in chunks without storing the full attention matrix.

### 3. Triton (The Language)

Triton is Python-like but compiles to GPU code. Key concepts:

```python
@triton.jit
def kernel(X_ptr, Y_ptr, N: tl.constexpr):
    # Each "program" runs on one SM (Streaming Multiprocessor)
    pid = tl.program_id(0)  # Which block am I?

    # Load a block of data
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(X_ptr + offs, mask=offs < N)

    # Compute
    y = x * 2

    # Store
    tl.store(Y_ptr + offs, y, mask=offs < N)
```

### 4. H100 Architecture (What We're Targeting)

```
H100 (Hopper):
├── 132 SMs (Streaming Multiprocessors)
├── 80GB HBM3 @ 3.35 TB/s bandwidth
├── 50MB L2 cache
├── Per SM:
│   ├── 256KB shared memory (SRAM)
│   ├── 4 Tensor Cores (for matrix multiply)
│   └── 128 CUDA cores
└── TensorCore specs:
    └── bf16: 1979 TFLOPS (with sparsity: 3958)
```

**Key insight**: TensorCores do 16x16 matrix multiply in one cycle. Our block sizes must be multiples of 16.

---

## Current State

```
Benchmark Results (H100):
┌──────────┬────────────┬────────────┬───────┐
│ Seq Len  │ Our Kernel │ Baseline   │ Ratio │
├──────────┼────────────┼────────────┼───────┤
│ 256      │ 0.459 ms   │ 0.014 ms   │ 0.03x │ ← We're 33x slower
│ 1024     │ 0.441 ms   │ 0.022 ms   │ 0.05x │ ← We're 20x slower
│ 4096     │ 0.536 ms   │ 0.229 ms   │ 0.43x │ ← We're 2.3x slower
│ 16384    │ 5.271 ms   │ 3.619 ms   │ 0.69x │ ← We're 1.5x slower
└──────────┴────────────┴────────────┴───────┘
```

**Observation**: Gap shrinks at longer sequences. This suggests launch overhead and block size issues.

---

## What We Need to Profile

### 1. Kernel Launch Overhead
```bash
# Use Nsight Systems to see timeline
nsys profile python benchmark.py
```
Questions:
- How much time in kernel launch vs actual compute?
- Are we launching too many small kernels?

### 2. Memory Bandwidth Utilization
```bash
# Use Nsight Compute for detailed metrics
ncu --set full python benchmark.py
```
Metrics to check:
- `dram_throughput` - Are we hitting 3.35 TB/s?
- `sm_efficiency` - Are all 132 SMs busy?
- `achieved_occupancy` - How full are the SMs?

### 3. TensorCore Utilization
```
ncu --metrics sm__inst_executed_pipe_tensor python benchmark.py
```
- Are TensorCores being used for `tl.dot()`?
- If not, block sizes or dtypes are wrong

### 4. Block Size Sweep
Test: BLOCK_M x BLOCK_N combinations
```
[32x32, 64x64, 128x64, 128x128, 256x64]
```
H100 sweet spot is usually 128x64 or 128x128.

---

## Optimization Roadmap

### Phase 1: Match Baseline (Current Focus)
1. **Fix block sizes** - Try BLOCK_M=128, BLOCK_N=64
2. **Remove dtype conversions** - Keep everything in bf16
3. **Tune num_warps** - H100 likes 4-8 warps per block
4. **Add autotuning** - Let Triton find best config

### Phase 2: Beat Baseline via Fusion
Once we match baseline attention, we win by fusing:
```
Current (separate kernels):
  Linear(QKV) → RoPE → Attention → Linear(out)
  ↓            ↓       ↓           ↓
  4 kernel launches, 4 HBM round-trips

Fused (one kernel):
  ┌─────────────────────────────────────┐
  │ Load X → QKV proj → RoPE → Attn →  │
  │ Out proj → Store                    │
  └─────────────────────────────────────┘
  1 kernel launch, 1 HBM round-trip
```

### Phase 3: Qwen3-VL Specific Optimizations
1. **Window attention** - Qwen3-VL uses local windows for efficiency
2. **3D mRoPE** - Position encoding across (time, height, width)
3. **Spatial merge** - Qwen3-VL downsamples 2x2 patches

---

## Files Structure

```
triton_kernels/
├── vision_attention.py   # Core Triton kernels
│   ├── _fused_attention_kernel    # FlashAttention implementation
│   ├── _window_attention_kernel   # Window-based variant
│   └── TritonVisionAttention      # PyTorch module wrapper
├── modal_app.py          # Modal deployment (H100 access)
├── PLAN.md              # This file
└── (future)
    ├── spatial_merge.py  # 2x2 patch merging kernel
    ├── rope_3d.py        # Fused 3D rotary embeddings
    └── benchmark.py      # Detailed profiling
```

---

## Next Steps (Priority Order)

1. **Add Triton autotuning** to find optimal block sizes
2. **Profile with ncu** to identify bottleneck
3. **Implement fused QKV+RoPE** kernel
4. **Add spatial merge kernel** for Qwen3-VL patch merging
5. **Benchmark full vision encoder** end-to-end

---

## Resources

- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - The algorithm we're implementing
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/) - Official docs
- [H100 Whitepaper](https://resources.nvidia.com/en-us-tensor-core) - Architecture details
- [Qwen3-VL Paper](https://arxiv.org/abs/2409.12191) - Model architecture
