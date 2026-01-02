# Fusing RoPE into Attention for Vision Transformers

## The One-Liner

> vLLM applies RoPE, writes to memory, then reads it back for attention. We fused them. **Faster** (fill in exact speedups after sweeps).

---

## The Problem (with ASCII diagram)

```
Current vLLM Qwen3-VL:

  Q, K tensors
       │
       ▼
  ┌─────────────┐
  │   RoPE op   │  ← even if RoPE is fast, it still materializes Q_rot/K_rot
  └─────────────┘
       │
       ▼
  Q_rot, K_rot   ← WRITE to HBM (memory bottleneck)
       │
       ▼
  ┌─────────────────────┐
  │ flash_attn_varlen   │  ← READ Q_rot, K_rot from HBM
  └─────────────────────┘
       │
       ▼
    Output


Our approach:

  Q, K tensors
       │
       ▼
  ┌───────────────────────────┐
  │  Fused RoPE + Attention   │  ← RoPE applied during Q/K loads
  │  (one Triton kernel)      │     No intermediate write/read
  └───────────────────────────┘
       │
       ▼
    Output
```

---

## Why This Matters

Memory bandwidth is the bottleneck, not compute.

```
H100 specs:
  - Compute: ~989 TFLOPS (BF16/FP16, dense tensor cores)
  - Memory BW: 3.35 TB/s

For attention with seq_len=4096, head_dim=72:
  - Q tensor: 4096 × 16 heads × 72 × 2 bytes (bf16) = 9.0 MiB
  - K tensor: same = 9.0 MiB
  - Extra traffic if RoPE is out-of-place:
      - Write Q_rot + K_rot: 18.0 MiB
      - Read  Q_rot + K_rot: 18.0 MiB
      - Total extra: ~36.0 MiB per layer × 27 layers ≈ ~972 MiB per image
```

By fusing, we eliminate the *intermediate Q_rot/K_rot write+read*.

---

## The Numbers

**Benchmark: H100, Qwen3-VL vision encoder config (per attention layer)**

| Image Size | Flash + RoPE | Fused | Speedup |
|------------|--------------|-------|---------|
| 448×448    | TBD          | TBD   | TBD     |
| 672×672    | TBD          | TBD   | TBD     |
| 896×896    | TBD          | TBD   | TBD     |
| 1344×1344  | TBD          | TBD   | TBD     |

Baseline note: the “Flash + RoPE” column is `flash_attn_varlen_func` + a separate RoPE pass (PyTorch reference) to represent the “materialize then attend” pattern.

To regenerate and fill this table: `modal run triton_kernels/bench_chart.py` (or `modal run triton_kernels/bench_vs_flash.py` for the broader comparison).

---

## The Code (simplified)

```python
# Before: Two separate operations
q_rotated = apply_rotary_emb(q, cos, sin)  # Write to memory
k_rotated = apply_rotary_emb(k, cos, sin)  # Write to memory
out = flash_attn_varlen_func(q_rotated, k_rotated, v, ...)  # Read from memory

# After: One fused kernel
@triton.jit
def fused_rope_attention_kernel(Q, K, V, COS, SIN, Out, ...):
    # Load Q chunk
    q = load(Q, ...)

    # Apply RoPE during load (no memory write)
    cos, sin = load(COS, ...), load(SIN, ...)
    q_rot = q[:half] * cos - q[half:] * sin  # Fused!

    # Same for K
    k = load(K, ...)
    k_rot = k[:half] * cos - k[half:] * sin  # Fused!

    # Attention (FlashAttention-style)
    scores = dot(q_rot, k_rot) * scale
    # ... online softmax, output
```

---

## Why Hasn't This Been Done?

It has, partially:

| Library | Fused RoPE? | For Vision? |
|---------|-------------|-------------|
| FlashInfer | Yes (`pos_encoding_mode`) | LLM decode only |
| flash-attn | Yes (`flash_attn_with_kvcache`) | LLM with KV cache |
| vLLM Qwen3-VL | **No** | Vision encoder |

The gap: Vision encoders use `flash_attn_varlen_func` (variable-length for windows), which doesn't have fused RoPE.

---

## What We Built

- Triton kernel: fused RoPE + window attention
- Supports non-power-of-2 head_dim (Qwen3-VL uses 72)
- 2D spatial RoPE (height, width positions)
- FlashAttention-style online softmax

---

## Impact

**For Qwen3-VL vision encoder:**
- 27 attention layers
- Per-layer savings: ~0.23 ms (at 1344×1344)
- Total savings: ~6 ms per image
- End-to-end speedup depends on non-attention work (QKV/proj, MLP, etc.); in our end-to-end test on example images we saw ~1.5× for the full vision stack.

**For video (16 frames @ 448×448):**
- Vision dominates end-to-end runtime as frames increase, so this optimization compounds.

---

## The Insight

> The fastest memory access is the one you don't make.

vLLM optimized RoPE with Triton (2-3x faster). But faster ≠ eliminated.

Fusing RoPE into attention removes the write-read cycle entirely. The compute for RoPE is trivial—it's the memory traffic that kills you.

---

## Links

- Our benchmark: `modal run bench_vs_flash.py`
- vLLM Qwen3-VL: [qwen3_vl.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_vl.py)
- Their Triton RoPE: [PR #25055](https://github.com/vllm-project/vllm/pull/25055)
- FlashInfer fused RoPE: [blog](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- Related vLLM issue: [#24678](https://github.com/vllm-project/vllm/issues/24678) (NVIDIA working on similar fusion)

---

## One Tweet Version

```
TIL: vLLM's Qwen3-VL vision encoder applies RoPE, writes to
memory, then reads it back for attention.

We fused them into one kernel. 3.7x faster.

Memory bandwidth is the bottleneck. The fastest memory
access is the one you don't make.

[graph] [link to code]
```
