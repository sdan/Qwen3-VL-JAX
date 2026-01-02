"""Reference: Official Triton FlashAttention v2 Tutorial
Source: https://github.com/triton-lang/triton/blob/main/python/tutorials/06-fused-attention.py

Key patterns extracted for our Qwen3-VL implementation.
"""

import torch
import triton
import triton.language as tl

# =============================================================================
# AUTOTUNING CONFIGS - These are the magic numbers for H100
# =============================================================================

def get_autotune_configs():
    """
    Optimal configs for H100:
    - BLOCK_M: 64 or 128 (128 usually better for long sequences)
    - BLOCK_N: 32, 64, or 128 (64 is sweet spot)
    - num_warps: 4 or 8 (4 for smaller blocks, 8 for larger)
    - num_stages: 2-4 (pipeline stages for async loads)
    """
    return [
        # (BLOCK_M, BLOCK_N, num_warps, num_stages)
        (128, 64, 4, 2),   # Good default for H100
        (128, 128, 8, 2),  # For very long sequences
        (64, 64, 4, 2),    # For shorter sequences
        (64, 32, 4, 2),    # Memory-constrained
    ]

# Best single config for H100 (from Triton tutorial):
BLOCK_M = 128
BLOCK_N = 64
NUM_WARPS = 4
NUM_STAGES = 2


# =============================================================================
# CORE ATTENTION KERNEL STRUCTURE
# =============================================================================

@triton.jit
def _attn_fwd_inner(
    acc, l_i, m_i,  # Accumulators (in registers)
    q,              # Query block [BLOCK_M, HEAD_DIM] - stays in SRAM
    K_block_ptr, V_block_ptr,  # Pointers to K, V
    start_m,        # Which M block we're processing
    qk_scale,       # 1/sqrt(d)
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,  # 1 = causal mask, 2 = no mask
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
):
    """
    Inner loop: iterate over K,V blocks and accumulate attention output.

    This is the FlashAttention algorithm core:
    1. Load K block
    2. Compute QK^T (scaled)
    3. Apply mask if causal
    4. Online softmax update (m_i, l_i)
    5. Load V block
    6. Accumulate P @ V into acc
    7. Advance to next K,V block
    """
    # Determine iteration range based on causal vs non-causal
    if STAGE == 1:  # Causal: only attend to past
        lo, hi = 0, start_m * BLOCK_M
    else:  # Full attention
        lo, hi = 0, N_CTX

    # Advance pointers to starting position
    K_block_ptr = tl.advance(K_block_ptr, (0, lo))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    # Main loop over K,V blocks
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block [BLOCK_N, HEAD_DIM]
        k = tl.load(K_block_ptr)

        # QK^T: [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, tl.trans(k)) * qk_scale

        # Causal mask (if needed)
        if STAGE == 1:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = tl.where(mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)  # exp2 is faster than exp
        l_ij = tl.sum(p, 1)

        # Rescale previous accumulator
        alpha = tl.math.exp2(m_i - m_ij)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + l_ij
        m_i = m_ij

        # Load V and accumulate
        v = tl.load(V_block_ptr)
        acc = tl.dot(p.to(v.dtype), v, acc)

        # Advance pointers
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    return acc, l_i, m_i


# =============================================================================
# KEY OPTIMIZATIONS TO APPLY
# =============================================================================

"""
1. USE exp2 instead of exp:
   - exp2(x) = 2^x is faster on GPU
   - Adjust scale: qk_scale = log2(e) / sqrt(d)

2. BLOCK POINTERS (tl.make_block_ptr):
   - More efficient than manual indexing
   - Enables automatic bounds checking

3. PERSISTENT KERNELS (for short sequences):
   - Keep Q in registers across all K,V blocks
   - Reduces register spilling

4. SPLIT-K (for very long sequences):
   - Multiple thread blocks work on same output
   - Requires atomic reduction at end

5. FP8 (FlashAttention-3):
   - H100 has native FP8 TensorCores
   - 2x throughput vs bf16
   - Requires careful scaling

6. ASYNC LOADS (num_stages > 1):
   - Overlap memory loads with compute
   - H100 supports up to 4 stages
"""


# =============================================================================
# QWEN3-VL SPECIFIC ADDITIONS NEEDED
# =============================================================================

"""
For Qwen3-VL vision encoder, we need to add:

1. WINDOW ATTENTION:
   - Each window processed independently
   - cu_seqlens array marks window boundaries
   - Enables parallelism across windows

2. 3D mRoPE (Multimodal Rotary Position Embeddings):
   - Position encoding for (time, height, width)
   - Applied to Q and K before attention
   - Can be fused into attention kernel

3. SPATIAL MERGE:
   - Downsample 2x2 patches after attention
   - Simple avg/concat operation
   - Good fusion candidate

Fusion order:
  x → QKV_proj → split → apply_3d_rope(Q,K) → window_attention → merge → out_proj
  └─────────────────────────────────────────────────────────────────────────────┘
                              ONE FUSED KERNEL
"""
