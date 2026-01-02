"""Sampling, image preprocessing, and inference for Qwen3-VL

All inference logic in one file: image prep, tokenization, sampling, VLM inputs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Union, Any, Tuple

import jax, jax.numpy as jnp
import numpy as np

# Import from consolidated model
from model import KVCache, VisionEmbeddings, build_mrope, build_text_rope

# ============================================================================
# Image preprocessing
# ============================================================================

# Qwen3-VL uses simple normalization: (x - 0.5) / 0.5 = 2x - 1, mapping [0,1] to [-1,1]
IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
DEFAULT_MIN_PIXELS = 56 * 56
DEFAULT_MAX_PIXELS = 12845056


def smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int) -> tuple[int, int]:
    """Resize image to align with patch grid while respecting pixel bounds"""
    if height <= 0 or width <= 0:
        raise ValueError("Image dimensions must be positive")
    if min(height, width) == 0 or (max(height, width) / min(height, width)) > 200:
        raise ValueError("Aspect ratio must be < 200")

    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    area = h_bar * w_bar
    if area > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif area < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = max(factor, math.ceil(height * beta / factor) * factor)
        w_bar = max(factor, math.ceil(width * beta / factor) * factor)

    return int(h_bar), int(w_bar)


def preprocess_image(image: Union[str, Any], patch_size: int, spatial_merge_size: int,
                    temporal_patch_size: int, min_pixels: int = DEFAULT_MIN_PIXELS,
                    max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert image to Qwen3-VL format: patches + grid_thw

    Accepts filesystem path, PIL.Image, or numpy array HWC in [0,255] or [0,1].
    Returns (pixel_values, grid_thw) where pixel_values is [N_tokens, patch_volume].
    """
    import os
    from PIL import Image

    # Load/convert to PIL
    if isinstance(image, str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
        with Image.open(image) as img:
            pil_img = img.convert("RGB")
    elif hasattr(image, "convert"):
        pil_img = image.convert("RGB")
    else:
        arr = np.asarray(image)
        if arr.ndim != 3 or arr.shape[2] not in (3, 4):
            raise ValueError("Expected HWC array with 3 or 4 channels")
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        pil_img = Image.fromarray(arr.astype(np.uint8), mode="RGB")

    width, height = pil_img.size
    factor = patch_size * spatial_merge_size
    new_h, new_w = smart_resize(height, width, factor, min_pixels, max_pixels)

    if (new_w, new_h) != (width, height):
        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Normalize [0,255] -> [-1, 1]
    image_np = np.asarray(pil_img, dtype=np.float32) / 255.0
    image_np = (image_np - IMAGE_MEAN) / IMAGE_STD

    # CHW + temporal axis
    image_np = np.transpose(image_np, (2, 0, 1))[None, ...]  # (1, C, H, W)

    # Pad temporal dimension
    if temporal_patch_size > 1:
        frames = image_np.shape[0]
        remainder = frames % temporal_patch_size
        if remainder != 0:
            pad = temporal_patch_size - remainder
            image_np = np.concatenate([image_np, np.repeat(image_np[-1:], pad, axis=0)], axis=0)

    frames, channel, new_h, new_w = image_np.shape
    grid_t = frames // temporal_patch_size
    grid_h = new_h // patch_size
    grid_w = new_w // patch_size

    if grid_h == 0 or grid_w == 0:
        raise ValueError("Image too small for patch size")
    if grid_h * patch_size != new_h or grid_w * patch_size != new_w:
        raise ValueError("Dimensions must be divisible by patch_size")
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError("Grid must be divisible by spatial_merge_size")

    # Reshape to patches: (T, H/merge, W/merge, merge, merge, C, patch_t, patch_h, patch_w)
    patches = image_np.reshape(
        grid_t, temporal_patch_size, channel,
        grid_h // spatial_merge_size, spatial_merge_size, patch_size,
        grid_w // spatial_merge_size, spatial_merge_size, patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)

    pixel_values = jnp.asarray(flatten, dtype=jnp.float32)
    grid_thw = jnp.asarray([[grid_t, grid_h, grid_w]], dtype=jnp.int32)
    return pixel_values, grid_thw


# ============================================================================
# Prompting and tokenization helpers
# ============================================================================

def decode_tokens(tokenizer, token_ids: List[int]) -> str:
    """Decode token IDs to text"""
    return tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def chat_prompt_with_image(num_vision_tokens: int, question: str) -> str:
    """Format a single-image chat prompt"""
    return (
        f"<|im_start|>user\n"
        f"<|vision_start|>{'<|image_pad|>' * num_vision_tokens}<|vision_end|>"
        f"{question}<|im_end|>\n<|im_start|>assistant\n"
    )


def chat_prompt_with_images(num_tokens_list: List[int], question: str) -> str:
    """Format a multi-image chat prompt"""
    vision_blocks = "".join(f"<|vision_start|>{'<|image_pad|>' * int(n)}<|vision_end|>"
                           for n in num_tokens_list)
    return f"<|im_start|>user\n{vision_blocks}{question}<|im_end|>\n<|im_start|>assistant\n"


def extract_assistant(full_text: str) -> str | None:
    """Extract assistant response from chat format"""
    start = "<|im_start|>assistant\n"
    end = "<|im_end|>"
    if start not in full_text:
        return None
    st = full_text.rfind(start) + len(start)
    ed = full_text.find(end, st)
    return full_text[st:ed if ed != -1 else len(full_text)]


def token_positions(tokens: jnp.ndarray, pad_id: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute token positions for text-only RoPE"""
    mask = (tokens != pad_id).astype(jnp.int32)
    positions = jnp.cumsum(mask, axis=-1) - 1
    positions = jnp.where(mask > 0, positions, 0)
    return positions, mask


# ============================================================================
# Top-k / top-p logits masking
# ============================================================================

def apply_top_p_logits(logits: jnp.ndarray, top_p: float) -> jnp.ndarray:
    """Exact nucleus sampling (slow, full vocab sort)"""
    if top_p is None or not (0.0 < float(top_p) < 1.0):
        return logits
    sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
    sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
    sorted_probs = jax.nn.softmax(sorted_logits.astype(jnp.float32), axis=-1)
    cumprobs = jnp.cumsum(sorted_probs, axis=-1)
    keep = cumprobs <= jnp.float32(top_p)
    keep = keep.at[:, :1].set(True)
    masked_sorted = jnp.where(keep, sorted_logits, jnp.float32(-1e9))

    def _unsort(masked, indices):
        out = jnp.full_like(masked, jnp.float32(-1e9))
        return out.at[indices].set(masked)

    return jax.vmap(_unsort)(masked_sorted, sorted_indices)


def mask_logits_topk_topp(logits: jnp.ndarray, top_k: int | None, top_p: float | None) -> jnp.ndarray:
    """Fast top-k + top-p masking (recommended)"""
    vocab = logits.shape[-1]
    use_topk = (top_k is not None) and (int(top_k) > 0)
    use_topp = (top_p is not None) and (0.0 < float(top_p) < 1.0)

    if not use_topk and not use_topp:
        return logits
    if not use_topk:
        return apply_top_p_logits(logits, top_p)

    k = min(int(top_k), vocab)
    top_vals, top_idx = jax.lax.top_k(logits, k)

    if use_topp:
        probs = jax.nn.softmax(top_vals.astype(jnp.float32), axis=-1)
        cumprobs = jnp.cumsum(probs, axis=-1)
        keep = cumprobs <= jnp.float32(top_p)
        keep = keep.at[:, :1].set(True)
        masked_vals = jnp.where(keep, top_vals, jnp.float32(-1e9))
    else:
        masked_vals = top_vals

    out = jnp.full_like(logits, jnp.float32(-1e9))
    batch_idx = jnp.arange(logits.shape[0])[:, None]
    return out.at[batch_idx, top_idx].set(masked_vals)


# ============================================================================
# Typed sampling containers
# ============================================================================

@dataclass
class SamplingConfig:
    """Sampling hyperparameters"""
    temperature: float = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    eos_id: Optional[int] = None
    pad_id: int = 0
    max_new_tokens: int = 64


@dataclass
class VLMInputs:
    """Vision-language model inputs"""
    prompt_tokens: jnp.ndarray  # [batch, seq]
    vision: Union[VisionEmbeddings, jnp.ndarray]  # Vision features
    grid_thw: jnp.ndarray  # [batch, num_images, 3] or [num_images, 3]
    image_pad_id: int  # Token ID for <|image_pad|>
    vision_start_id: int  # Token ID for <|vision_start|>


@dataclass
class SampleResult:
    """Sampling output"""
    tokens: jnp.ndarray  # [batch, new_len]
    logprobs: Optional[jnp.ndarray]  # [batch, new_len]
    texts: List[str] = field(default_factory=list)


# ============================================================================
# Core sampling logic
# ============================================================================

@dataclass
class _RopeSpec:
    """Internal spec for RoPE computation"""
    rope_section: tuple
    rope_theta: float
    rope_scaling_type: Optional[str]
    rope_scaling_factor: Optional[float]
    dtype: jnp.dtype
    num_layers: int
    num_kv_heads: int
    head_dim: int
    mrope_interleaved: bool


def _rope_spec_from_model(model) -> _RopeSpec:
    """Extract RoPE config from model"""
    text_spec = model.spec.text
    return _RopeSpec(
        rope_section=tuple(text_spec.rope_section),
        rope_theta=float(text_spec.rope_theta),
        rope_scaling_type=getattr(text_spec, "rope_scaling_type", None),
        rope_scaling_factor=getattr(text_spec, "rope_scaling_factor", None),
        dtype=model.dtype,
        num_layers=int(text_spec.num_hidden_layers),
        num_kv_heads=int(text_spec.num_key_value_heads),
        head_dim=int(text_spec.head_dim),
        mrope_interleaved=bool(getattr(text_spec, "mrope_interleaved", False)),
    )


def _init_cache(spec: _RopeSpec, batch: int, max_len: int) -> KVCache:
    """Initialize empty KV cache"""
    return KVCache.init(batch, spec.num_layers, spec.num_kv_heads, spec.head_dim, max_len, spec.dtype)


def _prefill_text(model, params, tokens: jnp.ndarray, pad_id: int, spec: _RopeSpec,
                 max_cache_len: Optional[int]) -> Tuple[KVCache, jnp.ndarray]:
    """Prefill cache for text-only inputs"""
    if tokens.ndim != 2:
        raise ValueError("tokens must be [batch, seq]")

    positions, mask = token_positions(tokens, pad_id)
    cos, sin = build_text_rope(positions, spec.rope_section, spec.rope_theta, spec.dtype,
                               rope_scaling_type=spec.rope_scaling_type,
                               rope_scaling_factor=spec.rope_scaling_factor,
                               mrope_interleaved=spec.mrope_interleaved)
    cache = _init_cache(spec, tokens.shape[0], int(max_cache_len or tokens.shape[1]))

    def _prefill(params, tokens, cos, sin, mask, cache):
        _, cache_out = model.apply({"params": params}, tokens, cos, sin, mask=mask, cache=cache,
                                  method=model.forward_text)
        return cache_out
    _prefill = jax.jit(_prefill, donate_argnames=['cache'])

    cache_out = _prefill(params, tokens, cos, sin, mask, cache)
    rope_deltas = jnp.zeros((tokens.shape[0], 1), dtype=jnp.int32)
    return cache_out, rope_deltas


def _prefill_vlm(model, params, tokens: jnp.ndarray, vision: Union[VisionEmbeddings, jnp.ndarray],
                grid_thw: jnp.ndarray, image_pad_id: int, vision_start_id: int, pad_id: int, spec: _RopeSpec,
                max_cache_len: Optional[int]) -> Tuple[jnp.ndarray, KVCache, jnp.ndarray]:
    """Prefill cache for vision-language inputs"""
    if tokens.ndim != 2:
        raise ValueError("tokens must be [batch, seq]")

    mask = (tokens != pad_id).astype(jnp.int32)
    batch = int(tokens.shape[0])

    # Normalize grid to [B, N, 3]
    grid_thw = jnp.asarray(grid_thw, dtype=jnp.int32)
    if grid_thw.ndim == 2:
        if grid_thw.shape[0] == 1 and batch > 1:
            grid_thw = jnp.tile(grid_thw[None, ...], (batch, 1, 1))
        elif grid_thw.shape[0] == batch and grid_thw.shape[1] == 3:
            grid_thw = grid_thw[:, None, :]
    elif grid_thw.ndim == 3 and grid_thw.shape[0] == 1 and batch > 1:
        grid_thw = jnp.tile(grid_thw, (batch, 1, 1))

    # Get mRoPE indices for Qwen3-VL
    from model import get_rope_index
    pos3, deltas = get_rope_index(
        spatial_merge_size=model.spec.vision.spatial_merge_size,
        input_ids=tokens,
        image_grid_thw=grid_thw,
        attention_mask=mask,
        image_token_id=int(image_pad_id) if image_pad_id is not None else None,
        vision_start_id=int(vision_start_id) if vision_start_id is not None else None,
    )

    cos, sin = build_mrope(pos3, spec.rope_section, spec.rope_theta, spec.dtype,
                          rope_scaling_type=spec.rope_scaling_type,
                          rope_scaling_factor=spec.rope_scaling_factor,
                          mrope_interleaved=spec.mrope_interleaved)

    max_len = int(max_cache_len or tokens.shape[1])
    cache = _init_cache(spec, batch, max_len)

    # Cast vision features
    if isinstance(vision, VisionEmbeddings):
        vision_pack = vision.cast(spec.dtype)
    else:
        vision_arr = jnp.asarray(vision, dtype=spec.dtype)
        vision_pack = VisionEmbeddings(tokens=vision_arr, deepstack=())

    def _prefill(params, tokens, vision_pack, image_pad_id, cos, sin, mask, cache):
        logits, cache_out = model.apply({"params": params}, tokens, vision_pack, image_pad_id,
                                        cos, sin, mask=mask, cache=cache, method=model.forward_vlm)
        return logits, cache_out
    _prefill = jax.jit(_prefill, donate_argnames=['cache'])

    logits, cache = _prefill(params, tokens, vision_pack, image_pad_id, cos, sin, mask, cache)
    return logits, cache, deltas.astype(jnp.int32)


def _decode_loop(model, params, cache: KVCache, first_token: jnp.ndarray, steps: int,
                temperature: float, top_p: Optional[float], eos_id: Optional[int], top_k: Optional[int],
                rope_deltas: Optional[jnp.ndarray], rng: jax.Array, return_logprobs: bool
                ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Autoregressive decode loop"""
    if steps <= 0:
        empty = jnp.zeros((first_token.shape[0], 0), dtype=jnp.int32)
        return (empty, empty.astype(jnp.float32)) if return_logprobs else (empty, None)
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    temp = jnp.float32(temperature)
    eos_scalar = jnp.int32(eos_id if eos_id is not None else -1)
    has_eos = eos_scalar >= 0
    use_top_k = int(top_k) if top_k is not None else 0
    topp_val = float(top_p) if (top_p and 0.0 < float(top_p) < 1.0) else None

    def _scan_decode(params, offsets, cache_init, first_tok, rng_init):
        def _one_step(params, offsets, carry, _):
            cache_state, current_tok, rng_state, stopped = carry
            rng_state, step_key = jax.random.split(rng_state)
            step_mask = (jnp.logical_not(stopped)).astype(jnp.int32)[:, None]

            logits, cache_state = model.apply({"params": params}, current_tok, cache_state,
                                              offsets, step_mask, method=model.decode_step)
            logits = logits.astype(jnp.float32) / temp
            masked = mask_logits_topk_topp(logits, top_k=use_top_k, top_p=topp_val)
            next_token = jax.random.categorical(step_key, masked)

            if return_logprobs:
                log_probs = jax.nn.log_softmax(masked)
                gathered = log_probs[jnp.arange(log_probs.shape[0]), next_token]
            else:
                gathered = jnp.zeros((masked.shape[0],), dtype=jnp.float32)

            hit_eos = jnp.logical_and(has_eos, next_token == eos_scalar)
            stopped_new = jnp.logical_or(stopped, hit_eos)
            effective_next = jnp.where(jnp.logical_and(stopped, has_eos),
                                      jnp.broadcast_to(eos_scalar, next_token.shape), next_token)

            carry_out = (cache_state, effective_next.astype(jnp.int32), rng_state, stopped_new)
            y = (effective_next.astype(jnp.int32), gathered.astype(jnp.float32))
            return carry_out, y

        init_carry = (cache_init, first_tok.astype(jnp.int32), rng_init,
                     jnp.zeros_like(first_tok, dtype=jnp.bool_))
        carry_out, ys = jax.lax.scan(lambda c, _: _one_step(params, offsets, c, _),
                                     init_carry, xs=None, length=int(steps))
        tokens_seq, logprobs_seq = ys
        return tokens_seq.transpose(1, 0), logprobs_seq.transpose(1, 0)
    _scan_decode = jax.jit(_scan_decode, donate_argnames=['cache_init'])

    offsets = jnp.asarray(rope_deltas if rope_deltas is not None
                         else jnp.zeros((cache.lengths.shape[0], 1), dtype=jnp.int32))
    return _scan_decode(params, offsets, cache, first_token, rng)


def sample(model, params, inputs: Union[VLMInputs, jnp.ndarray, np.ndarray],
          cfg: SamplingConfig, rng: jax.Array, tokenizer=None, return_logprobs: bool = False
          ) -> SampleResult:
    """Main sampling entry point

    Accepts either:
    - VLMInputs for vision-language sampling
    - jnp.ndarray/np.ndarray [B, T] for text-only sampling

    Returns SampleResult with generated tokens, optional logprobs, and decoded texts.
    """
    spec = _rope_spec_from_model(model)

    # Determine input type and prefill
    if isinstance(inputs, VLMInputs):
        tokens = jnp.asarray(inputs.prompt_tokens, dtype=jnp.int32)
        _, cache, rope_deltas = _prefill_vlm(model, params, tokens, inputs.vision, inputs.grid_thw,
                                             inputs.image_pad_id, inputs.vision_start_id, cfg.pad_id, spec,
                                             max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens))
    else:
        tokens = jnp.asarray(inputs, dtype=jnp.int32)
        cache, rope_deltas = _prefill_text(model, params, tokens, cfg.pad_id, spec,
                                          max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens))

    # Get last non-pad token
    lengths = cache.lengths.astype(jnp.int32)
    last_idx = jnp.maximum(lengths - 1, 0)
    last_token = jnp.take_along_axis(tokens, last_idx[:, None], axis=1).squeeze(1)

    # Decode loop
    new_tokens, new_logprobs = _decode_loop(model, params, cache, last_token, int(cfg.max_new_tokens),
                                           float(cfg.temperature), cfg.top_p, cfg.eos_id, cfg.top_k,
                                           rope_deltas, rng, return_logprobs)

    # Decode texts
    texts = []
    if tokenizer is not None:
        try:
            for row in np.asarray(new_tokens):
                texts.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))
        except Exception:
            texts = [""] * int(new_tokens.shape[0])

    return SampleResult(tokens=new_tokens, logprobs=new_logprobs, texts=texts)


def sample_streaming(model, params, inputs: Union[VLMInputs, jnp.ndarray, np.ndarray],
                    cfg: SamplingConfig, rng: jax.Array, tokenizer=None, return_logprobs: bool = False):
    """Streaming sampling that yields tokens one at a time

    Accepts either:
    - VLMInputs for vision-language sampling
    - jnp.ndarray/np.ndarray [B, T] for text-only sampling

    Yields tuples of (token, text, logprob) for each generated token.
    Note: Only supports batch_size=1 for streaming.
    """
    spec = _rope_spec_from_model(model)

    # Determine input type and prefill
    if isinstance(inputs, VLMInputs):
        tokens = jnp.asarray(inputs.prompt_tokens, dtype=jnp.int32)
        if tokens.shape[0] != 1:
            raise ValueError("Streaming only supports batch_size=1")
        _, cache, rope_deltas = _prefill_vlm(model, params, tokens, inputs.vision, inputs.grid_thw,
                                             inputs.image_pad_id, inputs.vision_start_id, cfg.pad_id, spec,
                                             max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens))
    else:
        tokens = jnp.asarray(inputs, dtype=jnp.int32)
        if tokens.shape[0] != 1:
            raise ValueError("Streaming only supports batch_size=1")
        cache, rope_deltas = _prefill_text(model, params, tokens, cfg.pad_id, spec,
                                          max_cache_len=int(tokens.shape[1] + cfg.max_new_tokens))

    # Get last non-pad token
    lengths = cache.lengths.astype(jnp.int32)
    last_idx = jnp.maximum(lengths - 1, 0)
    current_token = jnp.take_along_axis(tokens, last_idx[:, None], axis=1).squeeze(1)

    # Setup parameters
    temp = jnp.float32(cfg.temperature)
    eos_scalar = jnp.int32(cfg.eos_id if cfg.eos_id is not None else -1)
    has_eos = eos_scalar >= 0
    use_top_k = int(cfg.top_k) if cfg.top_k is not None else 0
    topp_val = float(cfg.top_p) if (cfg.top_p and 0.0 < float(cfg.top_p) < 1.0) else None
    offsets = jnp.asarray(rope_deltas if rope_deltas is not None
                         else jnp.zeros((1, 1), dtype=jnp.int32))

    # JIT-compile single decode step for performance
    def _single_step(params, token, cache_state, offsets, rng_key):
        step_mask = jnp.ones((1, 1), dtype=jnp.int32)
        logits, cache_new = model.apply({"params": params}, token, cache_state,
                                        offsets, step_mask, method=model.decode_step)
        logits = logits.astype(jnp.float32) / temp
        masked = mask_logits_topk_topp(logits, top_k=use_top_k, top_p=topp_val)
        next_token = jax.random.categorical(rng_key, masked)

        if return_logprobs:
            log_probs = jax.nn.log_softmax(masked)
            logprob = log_probs[0, next_token[0]]
        else:
            logprob = jnp.float32(0.0)

        return next_token[0], logprob, cache_new
    _single_step = jax.jit(_single_step, donate_argnames=['cache_state'])

    # Stream tokens one by one
    stopped = False
    for step in range(int(cfg.max_new_tokens)):
        if stopped:
            break

        rng, step_key = jax.random.split(rng)
        next_token, logprob, cache = _single_step(params, current_token, cache, offsets, step_key)

        # Convert to Python int
        token_id = int(next_token)
        logprob_val = float(logprob) if return_logprobs else None

        # Check for EOS
        if has_eos and token_id == int(eos_scalar):
            stopped = True

        # Decode token to text
        text = ""
        if tokenizer is not None:
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False)
            except Exception:
                text = ""

        # Yield result
        yield (token_id, text, logprob_val)

        # Update current token for next iteration
        current_token = jnp.array([token_id], dtype=jnp.int32)


__all__ = [
    "SamplingConfig", "VLMInputs", "SampleResult",
    "preprocess_image", "chat_prompt_with_image", "chat_prompt_with_images",
    "extract_assistant", "sample", "sample_streaming",
]
