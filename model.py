"""Qwen3‑VL JAX model — all architecture in one file.

Small, readable, minimal abstraction. This file defines the full text
decoder and vision encoder with just enough structure to stay clear.
Shapes are kept explicit in docstrings where it matters.

Reference upstream config mapping for weights:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py
"""
from __future__ import annotations

import glob, json, re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import flax, flax.linen as nn, jax, jax.numpy as jnp
from flax import struct
from safetensors import safe_open

DType = jnp.dtype

# ============================================================================
# Data structures
# ============================================================================

@struct.dataclass
class VisionEmbeddings:
    """Container for vision tower outputs: tokens + optional deepstack features"""
    tokens: jax.Array
    deepstack: tuple[jax.Array, ...] = ()

    @classmethod
    def concatenate(cls, embeds: Sequence["VisionEmbeddings"]) -> "VisionEmbeddings":
        if not embeds:
            return cls(tokens=jnp.zeros((0, 0), dtype=jnp.float32), deepstack=())
        tokens = jnp.concatenate([e.tokens for e in embeds], axis=0)
        base_len = len(embeds[0].deepstack)
        for e in embeds[1:]:
            if len(e.deepstack) != base_len:
                raise ValueError("All VisionEmbeddings must have same deepstack length")
        deepstack = tuple(jnp.concatenate([e.deepstack[i] for e in embeds], axis=0)
                         for i in range(base_len))
        return cls(tokens=tokens, deepstack=deepstack)

    def cast(self, dtype: DType) -> "VisionEmbeddings":
        return VisionEmbeddings(
            tokens=self.tokens.astype(dtype),
            deepstack=tuple(f.astype(dtype) for f in self.deepstack)
        )

    def with_batch_dim(self, batch: int) -> "VisionEmbeddings":
        """Ensure batch dimension matches expected size"""
        tokens = self.tokens if self.tokens.ndim == 3 else self.tokens[None, ...]
        if tokens.shape[0] == 1 and batch > 1:
            tokens = jnp.tile(tokens, (batch, 1, 1))
        deepstack = []
        for feat in self.deepstack:
            if feat.ndim == 2:
                feat = feat[None, ...]
            if feat.shape[0] == 1 and batch > 1:
                feat = jnp.tile(feat, (batch, 1, 1))
            deepstack.append(feat)
        return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))


@dataclass
class TextBackboneSpec:
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rope_theta: float
    rope_section: Sequence[int]
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]
    mrope_interleaved: bool
    rms_norm_eps: float
    vocab_size: int


@dataclass
class VisionBackboneSpec:
    hidden_size: int
    out_hidden_size: int
    depth: int
    num_heads: int
    intermediate_size: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    window_size: int
    in_channels: int
    fullatt_block_indexes: Sequence[int]
    num_position_embeddings: Optional[int] = None
    deepstack_visual_indexes: Sequence[int] = ()


@dataclass
class Qwen3VLSpec:
    text: TextBackboneSpec
    vision: Optional[VisionBackboneSpec]
    pad_token_id: Optional[int]
    bos_token_id: Optional[int]
    eos_token_id: Optional[int]


class KVCache(flax.struct.PyTreeNode):
    """Stores cached keys/values for efficient autoregressive decoding"""
    keys: jax.Array      # [layers, batch, heads, max_len, head_dim]
    values: jax.Array
    lengths: jax.Array   # [batch] - current fill position per sample

    @classmethod
    def init(cls, batch: int, num_layers: int, num_heads: int, head_dim: int,
             max_len: int, dtype: DType) -> "KVCache":
        keys = jnp.zeros((num_layers, batch, num_heads, max_len, head_dim), dtype=dtype)
        values = jnp.zeros((num_layers, batch, num_heads, max_len, head_dim), dtype=dtype)
        lengths = jnp.zeros((batch,), dtype=jnp.int32)
        return cls(keys=keys, values=values, lengths=lengths)

    def update(self, layer_id: int, k: jax.Array, v: jax.Array,
               start_positions: jax.Array, chunk_lengths: jax.Array
               ) -> tuple[jax.Array, jax.Array, "KVCache"]:
        """Append new k/v for a layer, return full cached tensors"""
        def _update_one(cache_k, cache_v, new_k, new_v, start, chunk_len):
            mask = (jnp.arange(new_k.shape[1]) < chunk_len)[None, :, None]
            new_k, new_v = new_k * mask, new_v * mask
            updated_k = jax.lax.dynamic_update_slice(cache_k, new_k, (0, start, 0))
            updated_v = jax.lax.dynamic_update_slice(cache_v, new_v, (0, start, 0))
            return updated_k, updated_v

        layer_k, layer_v = self.keys[layer_id], self.values[layer_id]
        new_k, new_v = jax.vmap(_update_one)(layer_k, layer_v, k, v, start_positions, chunk_lengths)
        cache = self.replace(keys=self.keys.at[layer_id].set(new_k),
                            values=self.values.at[layer_id].set(new_v))
        return new_k, new_v, cache


# ============================================================================
# RoPE (Rotary Position Embeddings)
# ============================================================================

def rotate_half(x: jax.Array) -> jax.Array:
    """Rotate half the hidden dims of the input"""
    x1, x2 = jnp.split(x, 2, axis=-1)
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_multimodal_rotary_pos_emb(q: jax.Array, k: jax.Array, cos: jax.Array,
                                    sin: jax.Array, rope_section: Sequence[int],
                                    unsqueeze_dim: int = 1) -> tuple[jax.Array, jax.Array]:
    """Apply rotary embeddings to q/k with optional interleaved mRoPE.

    Args:
        q, k: [B, Hq or Hkv, T, Dh]
        cos, sin: cos/sin tables shaped for mRoPE sections
        rope_section: tuple of section sizes that sum to Dh/2 for non‑interleaved
        unsqueeze_dim: where to broadcast cos/sin over heads

    Returns:
        (q_embed, k_embed) with rotation applied to the rotary subspace. If
        interleaved mRoPE is used by config, only the rotary span is rotated
        and the pass‑through span is concatenated unchanged.
    """
    sections = tuple(int(x) for x in rope_section)
    total_dim = sum(sections)

    def _reorder(table: jax.Array) -> jax.Array:
        """Interleave sections for mRoPE - extract each section from its corresponding axis"""
        # table shape: [axes, B, T, 2*total_dim]
        # Extract the relevant section from each axis and concatenate
        chunks = []
        for axis_idx, sec in enumerate(sections):
            axis_table = table[axis_idx, ...]  # [B, T, 2*total_dim]
            offset = sum(sections[:axis_idx])
            # Extract from first half (the two halves are duplicates from build_mrope)
            chunk = axis_table[..., offset:offset+sec]
            chunks.append(chunk)
        # Concatenate: [B, T, total_dim]
        reordered = jnp.concatenate(chunks, axis=-1)
        # Duplicate to get 2*total_dim (for rotate_half to work correctly)
        return jnp.concatenate([reordered, reordered], axis=-1)

    cos_flat = _reorder(cos).astype(q.dtype)
    sin_flat = _reorder(sin).astype(q.dtype)
    cos_embed = jnp.expand_dims(cos_flat, axis=unsqueeze_dim)
    sin_embed = jnp.expand_dims(sin_flat, axis=unsqueeze_dim)

    rope_dim = total_dim * 2
    if rope_dim > q.shape[-1]:  # Interleaved: rotate only first part
        rotated_dim = total_dim
        q_rot, q_pass = q[..., :rotated_dim], q[..., rotated_dim:]
        k_rot, k_pass = k[..., :rotated_dim], k[..., rotated_dim:]
        cos_rot, sin_rot = cos_embed[..., :rotated_dim], sin_embed[..., :rotated_dim]
        q_embed = jnp.concatenate([q_rot * cos_rot + rotate_half(q_rot) * sin_rot, q_pass], axis=-1)
        k_embed = jnp.concatenate([k_rot * cos_rot + rotate_half(k_rot) * sin_rot, k_pass], axis=-1)
    else:  # Standard: rotate full head_dim
        q_embed = q * cos_embed + rotate_half(q) * sin_embed
        k_embed = k * cos_embed + rotate_half(k) * sin_embed
    return q_embed, k_embed


def build_mrope(position_ids_axes: jax.Array, rope_section: Sequence[int], rope_theta: float,
                dtype: DType = jnp.bfloat16, rope_scaling_type: Optional[str] = None,
                rope_scaling_factor: Optional[float] = None) -> tuple[jax.Array, jax.Array]:
    """Build 3D mRoPE tables for (t, h, w) axes.

    Args:
        position_ids_axes: [3, B, T] integer positions per axis
        rope_section: sizes for each axis subspace (sum is Dh/2 if non‑interleaved)
        rope_theta: RoPE base
        dtype: output dtype
        rope_scaling_type/factor: optional scaling

    Returns:
        (cos, sin) each shaped [3, B, T, 2*sum(rope_section)]
    """
    sections = tuple(int(x) for x in rope_section)
    pos = position_ids_axes.astype(jnp.float32)
    if rope_scaling_factor and rope_scaling_type in (None, "linear", "dynamic", "finetuned"):
        pos = pos / jnp.float32(rope_scaling_factor)

    total_dim = sum(sections)
    inv_freq = 1.0 / (rope_theta ** (jnp.arange(total_dim, dtype=jnp.float32) / total_dim))
    freqs = jnp.einsum("sbn,k->sbnk", pos, inv_freq, precision=jax.lax.Precision.HIGHEST)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    return jnp.cos(emb).astype(dtype), jnp.sin(emb).astype(dtype)


def build_text_rope(positions: jax.Array, rope_section: Sequence[int], rope_theta: float,
                   dtype: DType = jnp.bfloat16, rope_scaling_type: Optional[str] = None,
                   rope_scaling_factor: Optional[float] = None) -> tuple[jax.Array, jax.Array]:
    """Classic 1D RoPE for text tokens.

    positions: [B, T] -> we broadcast to 3 axes to share codepath with mRoPE.
    """
    axes = len(tuple(rope_section))
    pos_axes = jnp.broadcast_to(positions[None, ...], (axes,) + positions.shape)
    return build_mrope(pos_axes, rope_section, rope_theta, dtype, rope_scaling_type, rope_scaling_factor)


def get_rope_index(spatial_merge_size: int = 2, input_ids: Optional[jax.Array] = None,
                  image_grid_thw: Optional[jax.Array] = None,
                  attention_mask: Optional[jax.Array] = None) -> Tuple[jax.Array, jax.Array]:
    """Compute per‑token mRoPE indices for mixed text+vision sequences.

    Returns position_ids [3, B, T] and per‑batch offsets `deltas` to align
    decode‑time positions with prefill length. Text tokens get 1D positions
    broadcast to 3 axes; vision tokens use true (t,h,w) grid indices.
    """
    IMAGE_TOKEN_ID, VISION_START_ID = 151655, 151652
    batch, seq_len = (input_ids.shape if input_ids is not None
                     else (attention_mask.shape[0] if attention_mask is not None else (1, 1)))

    if input_ids is None or image_grid_thw is None:
        # Text-only fallback
        if attention_mask is not None:
            mask = attention_mask.astype(jnp.int32)
            positions = jnp.cumsum(mask, axis=-1) - 1
            positions = jnp.where(mask == 0, 1, positions)
            position_ids = jnp.tile(positions[None, ...], (3, 1, 1))
            deltas = position_ids.max(axis=0).max(axis=-1, keepdims=True) + 1 - seq_len
        else:
            position_ids = jnp.tile(jnp.arange(seq_len)[None, None, :], (3, batch, 1))
            deltas = jnp.zeros((batch, 1), dtype=jnp.int32)
        return position_ids, deltas

    # Vision + text: build position IDs that account for 3D spatial structure
    attention_mask = attention_mask if attention_mask is not None else jnp.ones_like(input_ids)
    position_ids = jnp.ones((3, batch, seq_len), dtype=input_ids.dtype)
    deltas = []
    image_idx = 0

    for i in range(batch):
        ids, mask = input_ids[i], attention_mask[i]
        valid = ids[mask == 1]
        vision_starts = jnp.argwhere(valid == VISION_START_ID).flatten()
        image_count = int((valid[vision_starts + 1] == IMAGE_TOKEN_ID).sum()) if vision_starts.size > 0 else 0
        tokens_list = valid.tolist()

        pos_list = []
        st, local_img_idx = 0, 0
        for _ in range(image_count):
            ed = tokens_list.index(IMAGE_TOKEN_ID, st) if IMAGE_TOKEN_ID in tokens_list[st:] else len(tokens_list)
            t, h, w = [int(x) for x in (image_grid_thw[i, local_img_idx] if image_grid_thw.ndim == 3
                                       else image_grid_thw[image_idx])]
            local_img_idx += 1
            image_idx += 1

            grid_h, grid_w = h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st
            st_idx = int(jnp.max(pos_list[-1])) + 1 if pos_list else 0

            # Text positions
            pos_list.append(jnp.tile(jnp.arange(text_len)[None, :], (3, 1)) + st_idx)

            # Image spatial positions
            t_idx = jnp.tile(jnp.arange(t)[:, None], (1, grid_h * grid_w)).reshape(-1)
            h_idx = jnp.tile(jnp.arange(grid_h)[None, :, None], (t, 1, grid_w)).reshape(-1)
            w_idx = jnp.tile(jnp.arange(grid_w)[None, None, :], (t, grid_h, 1)).reshape(-1)
            spatial = jnp.stack([t_idx, h_idx, w_idx], axis=0) + text_len + st_idx
            pos_list.append(spatial)
            st = ed + t * grid_h * grid_w

        # Remaining text
        if st < len(tokens_list):
            st_idx = int(jnp.max(pos_list[-1])) + 1 if pos_list else 0
            pos_list.append(jnp.tile(jnp.arange(len(tokens_list) - st)[None, :], (3, 1)) + st_idx)

        positions = jnp.concatenate(pos_list, axis=1) if pos_list else jnp.tile(jnp.arange(valid.shape[0])[None, :], (3, 1))
        sel = jnp.where(mask == 1)[0]
        position_ids = position_ids.at[:, i, sel].set(positions)
        deltas.append(jnp.array(int(jnp.max(positions)) + 1 - seq_len, dtype=jnp.int32))

    return position_ids, jnp.stack(deltas).reshape(batch, 1)


# ============================================================================
# Core layers
# ============================================================================

def rms_norm(x: jax.Array, gamma: jax.Array, eps: float) -> jax.Array:
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (gamma * x * jax.lax.rsqrt(variance + eps)).astype(x.dtype)


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gamma = self.param("weight", nn.initializers.ones, (self.hidden_size,), self.dtype)
        return rms_norm(x, gamma, self.eps)


class FeedForward(nn.Module):
    hidden_size: int
    intermediate_size: int
    dtype: DType = jnp.bfloat16
    use_bias: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        gate = nn.Dense(self.intermediate_size, use_bias=self.use_bias, dtype=self.dtype, name="gate_proj")(x)
        up = nn.Dense(self.intermediate_size, use_bias=self.use_bias, dtype=self.dtype, name="up_proj")(x)
        down = nn.Dense(self.hidden_size, use_bias=self.use_bias, dtype=self.dtype, name="down_proj")(nn.silu(gate) * up)
        return down


class MultiHeadAttention(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_section: Sequence[int]
    eps: float = 1e-6
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array,
                mask: Optional[jax.Array] = None, cache: Optional[KVCache] = None,
                layer_id: Optional[int] = None, update_lengths: bool = False
                ) -> tuple[jax.Array, Optional[KVCache]]:
        """Causal self‑attention with optional grouped‑query and KV cache.

        Args:
            x: [B, T, C]
            cos/sin: RoPE tables (text or mRoPE)
            mask: [B, T] 1 for valid tokens
            cache: KV cache for decoding; when provided, appends keys/values
            layer_id: which layer to write into in the cache
            update_lengths: if True, increments cache lengths by current chunk

        Returns:
            (out, cache) where out is [B, T, C]
        """
        # Project to q, k, v
        q = nn.Dense(self.num_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(self.num_kv_heads * self.head_dim, use_bias=True, dtype=self.dtype, name="v_proj")(x)

        batch, seqlen = q.shape[0], q.shape[1]
        q = q.reshape(batch, seqlen, self.num_heads, self.head_dim)
        k = k.reshape(batch, seqlen, self.num_kv_heads, self.head_dim)
        v = v.reshape(batch, seqlen, self.num_kv_heads, self.head_dim)

        # Apply QK normalization (Qwen-specific)
        q = RMSNorm(self.head_dim, self.eps, self.dtype, name="q_norm")(q)
        k = RMSNorm(self.head_dim, self.eps, self.dtype, name="k_norm")(k)

        # Transpose to [batch, heads, seq, dim] for attention
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Apply RoPE
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.rope_section)

        # Handle masking and cache
        if mask is not None:
            key_mask = mask.astype(jnp.float32)
            cache_lengths = key_mask.sum(axis=-1).astype(jnp.int32)
            k = k * key_mask[:, None, :, None].astype(k.dtype)
            v = v * key_mask[:, None, :, None].astype(v.dtype)
        else:
            cache_lengths = jnp.full((batch,), seqlen, dtype=jnp.int32)

        if cache is not None:
            k, v, cache = cache.update(layer_id, k, v, cache.lengths, cache_lengths)
            effective_lengths = cache.lengths + cache_lengths
            if update_lengths:
                cache = cache.replace(lengths=effective_lengths)
        else:
            effective_lengths = cache_lengths

        # Attention computation with grouped-query if needed
        history_mask = (jnp.arange(k.shape[2])[None, :] < effective_lengths[:, None]).astype(jnp.float32)
        if self.num_heads != self.num_kv_heads:
            repeats = self.num_heads // self.num_kv_heads
            q_grouped = q.reshape(batch, self.num_kv_heads, repeats, q.shape[2], self.head_dim)
            scores = jnp.einsum("bhgqd,bhkd->bhgqk", q_grouped.astype(jnp.float32), k.astype(jnp.float32)) * (self.head_dim ** -0.5)
            scores = scores.reshape(batch, self.num_heads, q.shape[2], k.shape[2])
        else:
            scores = jnp.einsum("bhqd,bhkd->bhqk", q.astype(jnp.float32), k.astype(jnp.float32)) * (self.head_dim ** -0.5)

        # Apply masks
        scores = scores + (1.0 - history_mask)[:, None, None, :] * -1e9
        if scores.shape[2] > 1:  # Causal mask
            causal = jnp.tril(jnp.ones((scores.shape[2], scores.shape[3]), dtype=jnp.float32))
            scores = scores + (1.0 - causal)[None, None, :, :] * -1e9

        weights = jax.nn.softmax(scores, axis=-1)

        # Apply weights to values
        if self.num_heads != self.num_kv_heads:
            weights_grouped = weights.reshape(batch, self.num_kv_heads, repeats, weights.shape[2], -1)
            out = jnp.einsum("bhgqk,bhkd->bhgqd", weights_grouped, v.astype(jnp.float32))
            out = out.reshape(batch, self.num_heads, weights.shape[2], self.head_dim).astype(self.dtype)
        else:
            out = jnp.einsum("bhqk,bhkd->bhqd", weights, v.astype(jnp.float32)).astype(self.dtype)

        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(batch, seqlen, -1)
        out = nn.Dense(self.hidden_size, use_bias=False, dtype=self.dtype, name="o_proj")(out)
        return out, cache


class DecoderBlock(nn.Module):
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    rope_section: Sequence[int]
    eps: float
    dtype: DType = jnp.bfloat16

    def setup(self):
        self.input_norm = RMSNorm(self.hidden_size, self.eps, self.dtype)
        self.post_norm = RMSNorm(self.hidden_size, self.eps, self.dtype)
        self.attn = MultiHeadAttention(self.hidden_size, self.num_heads, self.num_kv_heads,
                                       self.head_dim, self.rope_section, self.eps, self.dtype)
        self.mlp = FeedForward(self.hidden_size, self.intermediate_size, self.dtype)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array,
                mask: Optional[jax.Array], cache: Optional[KVCache], layer_id: int,
                update_lengths: bool = False) -> tuple[jax.Array, Optional[KVCache]]:
        attn_out, cache = self.attn(self.input_norm(x), cos, sin, mask, cache, layer_id, update_lengths)
        x = x + attn_out
        x = x + self.mlp(self.post_norm(x))
        return x, cache


# ----------------------------------------------------------------------------
# Aliases for readability (no behavior change)
# ----------------------------------------------------------------------------
MLP = FeedForward
CausalSelfAttention = MultiHeadAttention
Block = DecoderBlock


# ============================================================================
# Vision backbone
# ============================================================================

class VisionRotaryEmbedding(nn.Module):
    dim: int
    theta: float = 10000.0

    def __call__(self, seq_len: int) -> jax.Array:
        inv_freq = 1.0 / (self.theta ** (jnp.arange(self.dim, dtype=jnp.float32) / self.dim))
        return jnp.outer(jnp.arange(seq_len, dtype=jnp.float32), inv_freq)


class VisionPatchEmbed(nn.Module):
    embed_dim: int
    patch_volume: int
    dtype: DType = jnp.bfloat16

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        return nn.Dense(self.embed_dim, use_bias=False, dtype=self.dtype, name="proj")(x.astype(self.dtype))


class VisionAttention(nn.Module):
    hidden_size: int
    num_heads: int
    dtype: DType = jnp.bfloat16

    def setup(self):
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

    @nn.compact
    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, cu_seqlens: jax.Array) -> jax.Array:
        qkv = nn.Dense(3 * self.hidden_size, use_bias=True, dtype=self.dtype, name="qkv")(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        seq_len = x.shape[0]
        q = q.reshape(seq_len, self.num_heads, self.head_dim)
        k = k.reshape(seq_len, self.num_heads, self.head_dim)
        v = v.reshape(seq_len, self.num_heads, self.head_dim)

        # Apply RoPE
        cos = cos[:, :self.head_dim].astype(self.dtype)[:, None, :]
        sin = sin[:, :self.head_dim].astype(self.dtype)[:, None, :]
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin

        # Window-based attention
        num_windows = cu_seqlens.shape[0] - 1
        chunks = []
        for i in range(num_windows):
            start, end = int(cu_seqlens[i]), int(cu_seqlens[i + 1])
            if start >= end:
                continue
            q_w, k_w, v_w = q[start:end], k[start:end], v[start:end]
            q_w = jnp.transpose(q_w, (1, 0, 2))
            k_w = jnp.transpose(k_w, (1, 0, 2))
            v_w = jnp.transpose(v_w, (1, 0, 2))
            scores = jnp.einsum("hqd,hkd->hqk", q_w.astype(jnp.float32), k_w.astype(jnp.float32)) * self.scale
            weights = jax.nn.softmax(scores, axis=-1)
            out = jnp.einsum("hqk,hkd->hqd", weights, v_w.astype(jnp.float32)).astype(self.dtype)
            chunks.append(jnp.transpose(out, (1, 0, 2)))

        out = jnp.concatenate(chunks, axis=0).reshape(seq_len, self.hidden_size)
        return nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype, name="proj")(out)


class VisionBlock(nn.Module):
    spec: VisionBackboneSpec
    dtype: DType = jnp.bfloat16

    def setup(self):
        self.norm1 = RMSNorm(self.spec.hidden_size, 1e-6, self.dtype)
        self.norm2 = RMSNorm(self.spec.hidden_size, 1e-6, self.dtype)
        self.attn = VisionAttention(self.spec.hidden_size, self.spec.num_heads, self.dtype)
        self.mlp = FeedForward(self.spec.hidden_size, self.spec.intermediate_size, self.dtype, use_bias=True)

    def __call__(self, x: jax.Array, cos: jax.Array, sin: jax.Array, cu_seqlens: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x), cos, sin, cu_seqlens)
        x = x + self.mlp(self.norm2(x))
        return x


class VisionPatchMerger(nn.Module):
    context_dim: int
    out_dim: int
    spatial_merge_size: int
    use_postshuffle_norm: bool = False
    dtype: DType = jnp.bfloat16

    def setup(self):
        self.unit = self.spatial_merge_size ** 2
        self.hidden_size = self.context_dim * self.unit
        norm_dim = self.hidden_size if self.use_postshuffle_norm else self.context_dim
        self.norm = RMSNorm(norm_dim, 1e-6, self.dtype)
        self.linear1 = nn.Dense(self.hidden_size, use_bias=True, dtype=self.dtype)
        self.linear2 = nn.Dense(self.out_dim, use_bias=True, dtype=self.dtype)

    def __call__(self, x: jax.Array) -> jax.Array:
        if self.use_postshuffle_norm:
            x = x.reshape(-1, self.unit * self.context_dim)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.reshape(-1, self.unit * self.context_dim)
        x = nn.gelu(self.linear1(x))
        return self.linear2(x)


class Qwen3VisionTransformer(nn.Module):
    spec: VisionBackboneSpec
    dtype: DType = jnp.float32

    def setup(self):
        patch_vol = self.spec.in_channels * self.spec.temporal_patch_size * self.spec.patch_size ** 2
        self.patch_embed = VisionPatchEmbed(self.spec.hidden_size, patch_vol, self.dtype)
        rotary_dim = (self.spec.hidden_size // self.spec.num_heads) // 2
        self.rotary = VisionRotaryEmbedding(rotary_dim)
        self.blocks = [VisionBlock(self.spec, self.dtype) for _ in range(self.spec.depth)]
        self.merger = VisionPatchMerger(self.spec.hidden_size, self.spec.out_hidden_size,
                                       self.spec.spatial_merge_size, dtype=self.dtype)
        self.deepstack_visual_indexes = tuple(self.spec.deepstack_visual_indexes)
        self.deepstack_mergers = [VisionPatchMerger(self.spec.hidden_size, self.spec.out_hidden_size,
                                                    self.spec.spatial_merge_size, use_postshuffle_norm=True,
                                                    dtype=self.dtype) for _ in self.deepstack_visual_indexes]

    def _rot_pos_emb(self, grid_thw: jax.Array) -> jax.Array:
        """Compute rotary position embeddings for vision tokens"""
        pos_chunks = []
        for idx in range(grid_thw.shape[0]):
            t, h, w = grid_thw[idx]
            merge = self.spec.spatial_merge_size
            hpos = jnp.arange(h)[:, None].repeat(w, axis=1)
            wpos = jnp.arange(w)[None, :].repeat(h, axis=0)
            hpos = hpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
            wpos = wpos.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
            pos = jnp.stack([hpos, wpos], axis=-1)
            pos = jnp.tile(pos, (int(t), 1))
            pos_chunks.append(pos)
        pos_ids = jnp.concatenate(pos_chunks, axis=0)
        max_grid = int(jnp.max(grid_thw[:, 1:]))
        rotary_full = self.rotary(max_grid)
        return rotary_full[pos_ids].reshape(pos_ids.shape[0], -1)

    def _get_window_index(self, grid_thw: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Compute window-based attention indices"""
        window_indices, cu_seqlens = [], [0]
        window_id = 0
        vit_window = self.spec.window_size // self.spec.spatial_merge_size // self.spec.patch_size

        for t, h, w in grid_thw:
            gh, gw = h // self.spec.spatial_merge_size, w // self.spec.spatial_merge_size
            index = jnp.arange(t * gh * gw).reshape(t, gh, gw)
            pad_h = (vit_window - (gh % vit_window)) % vit_window
            pad_w = (vit_window - (gw % vit_window)) % vit_window
            num_h = (gh + pad_h) // vit_window
            num_w = (gw + pad_w) // vit_window

            index_pad = jnp.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
            index_pad = index_pad.reshape(t, num_h, vit_window, num_w, vit_window)
            index_pad = index_pad.transpose(0, 1, 3, 2, 4).reshape(t, num_h * num_w, vit_window, vit_window)

            seqlens = (index_pad != -100).sum(axis=(2, 3)).reshape(-1)
            index_new = index_pad.reshape(-1)
            index_new = index_new[index_new != -100]

            window_indices.append(index_new + window_id)
            cu_tmp = jnp.cumsum(seqlens) * (self.spec.spatial_merge_size ** 2) + cu_seqlens[-1]
            cu_seqlens.extend(list(cu_tmp))
            window_id += int(t * gh * gw)

        window_index = jnp.concatenate(window_indices, axis=0)
        cu_arr = jnp.array(cu_seqlens, dtype=jnp.int32)
        mask = jnp.concatenate([jnp.array([True]), cu_arr[1:] != cu_arr[:-1]])
        return window_index, cu_arr[mask]

    def __call__(self, pixel_values: jax.Array, grid_thw: jax.Array) -> tuple[jax.Array, tuple]:
        x = self.patch_embed(pixel_values)
        rotary_emb = self._rot_pos_emb(grid_thw)
        window_idx, cu_window = self._get_window_index(grid_thw)

        # Shuffle for window attention
        seq_len, merge_unit = x.shape[0], self.spec.spatial_merge_size ** 2
        x = x.reshape(seq_len // merge_unit, merge_unit, -1)[window_idx, :, :].reshape(seq_len, -1)
        rotary_emb = rotary_emb.reshape(seq_len // merge_unit, merge_unit, -1)[window_idx, :, :].reshape(seq_len, -1)

        # Duplicate for full head_dim
        emb = jnp.concatenate([rotary_emb, rotary_emb], axis=-1)
        cos, sin = jnp.cos(emb).astype(self.dtype), jnp.sin(emb).astype(self.dtype)

        # Cumulative sequence lengths for full attention blocks
        cu_seqlens = jnp.cumsum(jnp.array([int(t * h * w * merge_unit) for t, h, w in grid_thw]))
        cu_seqlens = jnp.concatenate([jnp.array([0]), cu_seqlens])

        # Forward through blocks
        fullatt_set = set(self.spec.fullatt_block_indexes)
        deepstack_feats = []
        for i, block in enumerate(self.blocks):
            cu = cu_seqlens if i in fullatt_set else cu_window
            x = block(x, cos, sin, cu)
            if i in self.deepstack_visual_indexes:
                feat = self.deepstack_mergers[len(deepstack_feats)](x)
                deepstack_feats.append(feat)

        # Merge and unshuffle
        x = self.merger(x)
        reverse_idx = jnp.argsort(window_idx)
        x = x[reverse_idx, :]
        deepstack_feats = [f[reverse_idx, :] for f in deepstack_feats]
        return x, tuple(deepstack_feats)


# ============================================================================
# Main VLM model
# ============================================================================

class Qwen3VLModel(nn.Module):
    spec: Qwen3VLSpec
    dtype: DType = jnp.bfloat16

    def setup(self):
        text = self.spec.text
        self.embed = nn.Embed(text.vocab_size, text.hidden_size,
                             embedding_init=nn.initializers.normal(stddev=0.02), dtype=self.dtype)
        self.layers = [DecoderBlock(text.hidden_size, text.num_attention_heads, text.num_key_value_heads,
                                    text.head_dim, text.intermediate_size, tuple(text.rope_section),
                                    text.rms_norm_eps, self.dtype) for _ in range(text.num_hidden_layers)]
        self.final_norm = RMSNorm(text.hidden_size, text.rms_norm_eps, self.dtype)
        self.lm_head = nn.Dense(text.vocab_size, use_bias=False, dtype=jnp.float32)
        self.visual = Qwen3VisionTransformer(self.spec.vision) if self.spec.vision else None

    @staticmethod
    def _apply_deepstack(hidden: jax.Array, visual_mask: Optional[jax.Array], features: jax.Array) -> jax.Array:
        """Add deepstack vision features to hidden states at vision token positions"""
        if visual_mask is None or features.size == 0:
            return hidden
        def _add(h, mask, feat):
            if feat.shape[0] == 0:
                return h
            idx = jnp.where(mask, size=feat.shape[0], fill_value=-1)[0]
            valid = idx >= 0
            idx = jnp.where(valid, idx, 0)
            updates = jnp.where(valid[:, None], feat.astype(h.dtype), jnp.zeros_like(feat, dtype=h.dtype))
            return h.at[idx].add(updates)
        return jax.vmap(_add)(hidden, visual_mask.astype(bool), features)

    def _decode_from_hidden(self, hidden: jax.Array, cos: jax.Array, sin: jax.Array,
                           mask: Optional[jax.Array] = None, cache: Optional[KVCache] = None,
                           visual_mask: Optional[jax.Array] = None,
                           deepstack: Optional[tuple] = None) -> tuple[jax.Array, Optional[KVCache]]:
        new_cache = cache
        deepstack = deepstack or ()
        for i, layer in enumerate(self.layers):
            hidden, new_cache = layer(hidden, cos, sin, mask, new_cache, i,
                                     update_lengths=(cache is not None and i == len(self.layers) - 1))
            if deepstack and i < len(deepstack) and visual_mask is not None:
                hidden = self._apply_deepstack(hidden, visual_mask, deepstack[i])
        hidden = self.final_norm(hidden)
        logits = self.lm_head(hidden.astype(jnp.float32))
        return logits, new_cache

    def forward_text(self, tokens: jax.Array, cos: jax.Array, sin: jax.Array,
                    mask: Optional[jax.Array] = None, cache: Optional[KVCache] = None
                    ) -> tuple[jax.Array, Optional[KVCache]]:
        hidden = self.embed(tokens)
        return self._decode_from_hidden(hidden, cos, sin, mask, cache)

    def forward_vlm(self, tokens: jax.Array, vision_embeds: Union[jax.Array, VisionEmbeddings],
                   image_pad_id: int, cos: jax.Array, sin: jax.Array,
                   mask: Optional[jax.Array] = None, cache: Optional[KVCache] = None
                   ) -> tuple[jax.Array, Optional[KVCache]]:
        hidden = self.embed(tokens)
        batch = hidden.shape[0]

        # Normalize vision inputs
        if isinstance(vision_embeds, VisionEmbeddings):
            vision_pack = vision_embeds.cast(self.dtype).with_batch_dim(batch)
        else:
            vision_arr = jnp.asarray(vision_embeds, dtype=self.dtype)
            if vision_arr.ndim == 2:
                vision_arr = vision_arr[None, ...]
            if vision_arr.shape[0] == 1 and batch > 1:
                vision_arr = jnp.tile(vision_arr, (batch, 1, 1))
            vision_pack = VisionEmbeddings(tokens=vision_arr, deepstack=())

        # Inject vision tokens at <|image_pad|> positions
        visual_mask = (tokens == jnp.int32(image_pad_id))
        def _inject(h, tok, vis):
            num_vis = vis.shape[0]
            pos = jnp.where(tok == jnp.int32(image_pad_id), size=num_vis, fill_value=-1)[0]
            valid = pos >= 0
            pos = jnp.where(valid, pos, 0)
            updates = jnp.where(valid[:, None], vis.astype(h.dtype), h[pos])
            return h.at[pos].set(updates)

        hidden = jax.vmap(_inject)(hidden, tokens, vision_pack.tokens)
        return self._decode_from_hidden(hidden, cos, sin, mask, cache, visual_mask, vision_pack.deepstack)

    def encode_vision(self, pixel_values: jax.Array, grid_thw: jax.Array) -> VisionEmbeddings:
        if self.visual is None:
            raise ValueError("Vision backbone not configured")
        tokens, deepstack = self.visual(pixel_values, grid_thw)
        return VisionEmbeddings(tokens=tokens, deepstack=tuple(deepstack))

    def decode_step(self, token: jax.Array, cache: KVCache, rope_deltas: Optional[jax.Array],
                   mask: Optional[jax.Array] = None) -> tuple[jax.Array, KVCache]:
        """Single autoregressive decoding step"""
        positions = cache.lengths[:, None]
        batch, axes = positions.shape[0], len(tuple(self.spec.text.rope_section))
        base_pos = jnp.broadcast_to(positions[None, :, :], (axes, batch, 1))
        offsets = rope_deltas.astype(jnp.int32)[None, :, :] if rope_deltas is not None else jnp.zeros((axes, batch, 1), dtype=jnp.int32)
        pos_axes = base_pos + offsets
        cos, sin = build_mrope(pos_axes, tuple(self.spec.text.rope_section), self.spec.text.rope_theta,
                              self.dtype, self.spec.text.rope_scaling_type, self.spec.text.rope_scaling_factor)
        mask = mask if mask is not None else jnp.ones((token.shape[0], 1), dtype=jnp.int32)
        logits, new_cache = self.forward_text(token[:, None], cos, sin, mask, cache)
        return logits[:, -1, :], new_cache

    def __call__(self, tokens: jax.Array, cos: jax.Array, sin: jax.Array,
                mask: Optional[jax.Array] = None, cache: Optional[KVCache] = None
                ) -> tuple[jax.Array, Optional[KVCache]]:
        return self.forward_text(tokens, cos, sin, mask, cache)


__all__ = [
    # Specs and containers
    "TextBackboneSpec", "VisionBackboneSpec", "Qwen3VLSpec", "KVCache", "VisionEmbeddings",
    # Core layers
    "RMSNorm", "FeedForward", "MLP", "MultiHeadAttention", "CausalSelfAttention", "DecoderBlock", "Block",
    # RoPE
    "build_text_rope", "build_mrope", "get_rope_index", "apply_multimodal_rotary_pos_emb",
    # Model entry points are defined later in file
]


# ============================================================================
# Model loading
# ============================================================================

def _load_hf_config(hf_dir: str) -> dict:
    with open(f"{hf_dir}/config.json") as f:
        return json.load(f)


def spec_from_config(cfg: dict) -> Qwen3VLSpec:
    """Parse HF config into our spec"""
    text_cfg = cfg.get("text_config", cfg)
    rope_cfg = text_cfg.get("rope_scaling", cfg.get("rope_scaling", {}))
    head_dim = text_cfg.get("head_dim") or text_cfg["hidden_size"] // text_cfg["num_attention_heads"]
    vision_cfg = cfg.get("vision_config")

    # Parse RoPE config
    rope_type = rope_cfg.get("type") if isinstance(rope_cfg, dict) else None
    rope_factor = rope_cfg.get("factor", rope_cfg.get("finetuned_factor")) if isinstance(rope_cfg, dict) else None
    rope_interleaved = bool(rope_cfg.get("mrope_interleaved", False)) if isinstance(rope_cfg, dict) else False
    raw_section = rope_cfg.get("mrope_section") if isinstance(rope_cfg, dict) else None

    if raw_section is None:
        if vision_cfg is not None:
            raise ValueError("Missing rope_scaling.mrope_section for vision model")
        rope_section = [head_dim // 2]
    else:
        rope_section = [int(x) for x in raw_section]
        if vision_cfg and len(rope_section) != 3:
            raise ValueError("mrope_section must have 3 entries for vision models")
        if not rope_interleaved and sum(rope_section) != head_dim // 2:
            raise ValueError(f"mrope_section sum must equal head_dim//2 ({head_dim // 2})")

    text = TextBackboneSpec(
        hidden_size=text_cfg["hidden_size"],
        num_attention_heads=text_cfg["num_attention_heads"],
        num_hidden_layers=text_cfg["num_hidden_layers"],
        num_key_value_heads=text_cfg["num_key_value_heads"],
        head_dim=head_dim,
        intermediate_size=text_cfg["intermediate_size"],
        rope_theta=text_cfg.get("rope_theta", cfg.get("rope_theta", 10000.0)),
        rope_section=rope_section,
        rope_scaling_type=rope_type,
        rope_scaling_factor=float(rope_factor) if rope_factor else None,
        mrope_interleaved=rope_interleaved,
        rms_norm_eps=text_cfg["rms_norm_eps"],
        vocab_size=text_cfg["vocab_size"],
    )

    vision = None
    if vision_cfg:
        patch_sz = vision_cfg.get("patch_size", vision_cfg.get("spatial_patch_size"))
        temp_patch = vision_cfg.get("temporal_patch_size", vision_cfg.get("temporal_patch", 1))
        window_sz = vision_cfg.get("window_size", patch_sz * vision_cfg.get("spatial_merge_size", 1))
        vision = VisionBackboneSpec(
            hidden_size=vision_cfg["hidden_size"],
            out_hidden_size=vision_cfg["out_hidden_size"],
            depth=vision_cfg["depth"],
            num_heads=vision_cfg["num_heads"],
            intermediate_size=vision_cfg["intermediate_size"],
            patch_size=patch_sz,
            temporal_patch_size=temp_patch,
            spatial_merge_size=vision_cfg["spatial_merge_size"],
            window_size=window_sz,
            in_channels=vision_cfg.get("in_channels", vision_cfg.get("in_chans", 3)),
            fullatt_block_indexes=vision_cfg.get("fullatt_block_indexes", vision_cfg.get("deepstack_visual_indexes", [])),
            num_position_embeddings=vision_cfg.get("num_position_embeddings"),
            deepstack_visual_indexes=tuple(vision_cfg.get("deepstack_visual_indexes", [])),
        )

    return Qwen3VLSpec(text=text, vision=vision, pad_token_id=cfg.get("pad_token_id"),
                      bos_token_id=cfg.get("bos_token_id"), eos_token_id=cfg.get("eos_token_id"))


# Regex rules: HF torch param names -> Flax tree paths
_TEXT_KEY_RULES = {
    r"model\.language_model\.(model\.)?embed_tokens\.weight": "embed/embedding",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.self_attn\.(q|k|v)_proj\.(weight|bias)": r"layers_\2/attn/\3_proj/\4",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.self_attn\.(q|k)_norm\.weight": r"layers_\2/attn/\3_norm/weight",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.self_attn\.o_proj\.weight": r"layers_\2/attn/o_proj/kernel",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.mlp\.(gate|up|down)_proj\.weight": r"layers_\2/mlp/\3_proj/kernel",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.input_layernorm\.weight": r"layers_\2/input_norm/weight",
    r"model\.language_model\.(model\.)?layers\.(\d+)\.post_attention_layernorm\.weight": r"layers_\2/post_norm/weight",
    r"model\.language_model\.(model\.)?norm\.weight": "final_norm/weight",
    r"lm_head\.weight": "lm_head/kernel",
    r"model\.(embed_tokens|layers|norm)\.": r"\1/",  # Fallback for text-only
}

_VISION_KEY_RULES = {
    r"(model\.)?visual\.blocks\.(\d+)\.(norm1|norm2)\.weight": r"visual/blocks_\2/\3/weight",
    r"(model\.)?visual\.blocks\.(\d+)\.attn\.(qkv|proj)\.(weight|bias)": r"visual/blocks_\2/attn/\3/\4",
    r"(model\.)?visual\.blocks\.(\d+)\.mlp\.(gate|up|down)_proj\.(weight|bias)": r"visual/blocks_\2/mlp/\3_proj/\4",
    r"(model\.)?visual\.merger\.ln_q\.weight": "visual/merger/norm/weight",
    r"(model\.)?visual\.merger\.mlp\.(0|2)\.(weight|bias)": r"visual/merger/linear\1/\2",
    r"(model\.)?visual\.deepstack_merger_list\.(\d+)\.norm\.weight": r"visual/deepstack_mergers_\2/norm/weight",
    r"(model\.)?visual\.deepstack_merger_list\.(\d+)\.linear_fc(1|2)\.(weight|bias)": r"visual/deepstack_mergers_\2/linear\3/\4",
}


def _torch_key_to_flax(key: str) -> Optional[str]:
    """Convert torch param name to Flax param path"""
    for rules in (_TEXT_KEY_RULES, _VISION_KEY_RULES):
        for pattern, target in rules.items():
            if re.match(pattern, key):
                flax_key = re.sub(pattern, target, key)
                # Fix kernel vs weight/bias naming
                if "weight" in key and "kernel" not in flax_key:
                    flax_key = flax_key.replace("weight", "kernel")
                return flax_key
    return None


def create_model_from_hf(hf_dir: str) -> tuple[Qwen3VLModel, dict]:
    """Load model from HuggingFace checkpoint"""
    cfg = _load_hf_config(hf_dir)
    spec = spec_from_config(cfg)
    model = Qwen3VLModel(spec)

    # Initialize params
    dummy_ids = jnp.zeros((1, 1), dtype=jnp.int32)
    axes = len(tuple(spec.text.rope_section))
    rope_dim = sum(int(x) for x in spec.text.rope_section) * 2
    dummy_cos = jnp.zeros((axes, 1, 1, rope_dim), dtype=model.dtype)
    dummy_sin = jnp.zeros((axes, 1, 1, rope_dim), dtype=model.dtype)
    params = flax.core.unfreeze(model.init(jax.random.PRNGKey(0), dummy_ids, dummy_cos, dummy_sin)["params"])

    if spec.vision:
        patch_vol = spec.vision.in_channels * spec.vision.temporal_patch_size * spec.vision.patch_size ** 2
        merge = spec.vision.spatial_merge_size
        dummy_pix = jnp.zeros((merge * merge, patch_vol), dtype=model.dtype)
        dummy_grid = jnp.array([[1, merge, merge]], dtype=jnp.int32)
        vision_params = model.init(jax.random.PRNGKey(1), dummy_pix, dummy_grid, method=model.encode_vision)["params"]
        params.update(flax.core.unfreeze(vision_params))

    # Load safetensors
    safetensor_paths = glob.glob(f"{hf_dir}/*.safetensors")
    if not safetensor_paths:
        raise FileNotFoundError(f"No safetensors in {hf_dir}")

    for path in safetensor_paths:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Special case: conv->linear for patch_embed
                if "patch_embed.proj.weight" in key:
                    tensor = f.get_tensor(key).float().numpy()
                    tensor = tensor.reshape(tensor.shape[0], -1).T
                    params.setdefault("visual", {}).setdefault("patch_embed", {}).setdefault("proj", {})["kernel"] = tensor
                    continue

                target = _torch_key_to_flax(key)
                if target is None:
                    continue

                tensor = f.get_tensor(key).float().numpy()
                keys = target.split("/")
                param_dict = params
                while len(keys) > 1:
                    param_dict = param_dict[keys.pop(0)]
                final_key = keys[0]
                if "kernel" in final_key:
                    tensor = tensor.T
                param_dict[final_key] = tensor

    # Handle tied embeddings
    if cfg.get("tie_word_embeddings", False) and "embed" in params:
        params.setdefault("lm_head", {})["kernel"] = params["embed"]["embedding"].T

    return model, flax.core.freeze(params)


def create_model_from_ckpt(ckpt_dir: str) -> tuple[Qwen3VLModel, dict]:
    """Load model from converted JAX checkpoint"""
    import pickle
    cfg = _load_hf_config(ckpt_dir)
    spec = spec_from_config(cfg)
    model = Qwen3VLModel(spec)
    params_path = f"{ckpt_dir}/params.pkl"
    try:
        with open(params_path, "rb") as f:
            params = pickle.load(f)["params"]
    except jax.errors.JaxRuntimeError as err:
        if "default_memory_space" not in str(err):
            raise
        cpu_devices = jax.devices("cpu")
        if not cpu_devices:
            raise
        # Fallback to loading params on CPU when Metal backend lacks memory space support.
        with open(params_path, "rb") as f, jax.default_device(cpu_devices[0]):
            params = pickle.load(f)["params"]
    return model, params


__all__ = [
    "Qwen3VLModel", "KVCache", "VisionEmbeddings", "Qwen3VLSpec",
    "build_text_rope", "build_mrope", "get_rope_index",
    "spec_from_config", "create_model_from_hf", "create_model_from_ckpt",
]
