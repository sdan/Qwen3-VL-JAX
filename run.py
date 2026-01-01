#!/usr/bin/env python3
"""Qwen3-VL inference

Usage:
    python run.py inference.image=path/to/image.jpg
    python run.py inference.image=image.jpg inference.prompt="What is in this image?"
    python run.py inference.image=image.jpg inference.stream=true
    python run.py inference.image=image.jpg sampling.temperature=0.8 sampling.max_new_tokens=256
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import chz
from transformers import AutoTokenizer
import time

from model import create_model_from_ckpt
from sample import (SamplingConfig, VLMInputs, sample, sample_streaming,
                   preprocess_image, chat_prompt_with_image, extract_assistant)
from utils import Config, setup_logger


def main(cfg: Config) -> None:
    # Validate image path
    if cfg.inference.image is None:
        raise ValueError("inference.image is required. Usage: python run.py inference.image=path/to/image.jpg")

    # Setup logger
    logger = setup_logger(__name__, level=cfg.logging.log_level,
                         log_file=cfg.logging.log_file, verbose=cfg.logging.verbose)

    # Device preference (only acts on CUDA; no MPS juggling)
    dev_pref = str(getattr(cfg.inference, "device", "auto")).lower()
    chosen = None
    if dev_pref == "cpu":
        cpus = jax.devices("cpu")
        if cpus:
            jax.config.update("jax_default_device", cpus[0])
            chosen = cpus[0]
    elif dev_pref in ("cuda", "gpu"):
        gpus = jax.devices("gpu")
        if gpus:
            jax.config.update("jax_default_device", gpus[0])
            chosen = gpus[0]
        else:
            logger.warning("Requested CUDA/GPU but none found; using auto")

    logger.info(f"Available JAX devices: {[device.platform for device in jax.devices()]}")
    if chosen is not None:
        logger.info(f"Default device: {chosen.platform}")

    # Load model
    logger.info(f"Loading model from {cfg.model.model_dir}")
    model, params = create_model_from_ckpt(cfg.model.model_dir)
    logger.info(f"Model: {model.spec.text.num_hidden_layers} layers, "
               f"{model.spec.text.hidden_size} hidden dim")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_dir, trust_remote_code=True)
    logger.info(f"Tokenizer: vocab size {len(tokenizer)}")

    # Preprocess image
    logger.info(f"Processing image: {cfg.inference.image}")
    pixel_values, grid_thw = preprocess_image(
        cfg.inference.image,
        patch_size=model.spec.vision.patch_size,
        spatial_merge_size=model.spec.vision.spatial_merge_size,
        temporal_patch_size=model.spec.vision.temporal_patch_size,
    )
    logger.debug(f"Image shape: {pixel_values.shape}, grid: {grid_thw}")

    # Encode vision
    logger.info("Encoding vision features...")
    vision_embeddings = model.apply(
        {"params": params},
        pixel_values,
        grid_thw,
        method=model.encode_vision,
    )
    # VisionEmbeddings.tokens is [seq, dim] or [batch, seq, dim]. We need the seq length.
    if vision_embeddings.tokens.ndim == 2:
        num_vision_tokens = int(vision_embeddings.tokens.shape[0])
    else:
        num_vision_tokens = int(vision_embeddings.tokens.shape[1])
    logger.info(f"Vision tokens: {num_vision_tokens}")

    # Format prompt
    formatted_prompt = chat_prompt_with_image(num_vision_tokens, cfg.inference.prompt)
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    logger.debug(f"Prompt length: {len(prompt_tokens)} tokens")

    # Get special token IDs
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
    # Prefer terminating the assistant turn on <|im_end|> for chat-style decoding
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_id = im_end_id if im_end_id is not None and int(im_end_id) >= 0 else tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Prepare inputs
    sampling_cfg = SamplingConfig(
        temperature=cfg.sampling.temperature,
        top_p=cfg.sampling.top_p,
        top_k=cfg.sampling.top_k,
        max_new_tokens=cfg.sampling.max_new_tokens,
        eos_id=eos_id,
        pad_id=pad_id,
    )

    inputs = VLMInputs(
        prompt_tokens=jnp.array([prompt_tokens], dtype=jnp.int32),
        vision=vision_embeddings,
        grid_thw=grid_thw,
        image_pad_id=image_pad_id,
        vision_start_id=vision_start_id,
    )

    # Generate
    rng = jax.random.PRNGKey(cfg.sampling.seed)
    t0 = time.perf_counter()

    if cfg.inference.stream:
        # Streaming: print tokens as they're generated
        logger.info("Streaming response...")
        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)

        all_tokens = []
        for token_id, text, _ in sample_streaming(model, params, inputs, sampling_cfg, rng, tokenizer=tokenizer):
            print(text, end='', flush=True)
            all_tokens.append(token_id)

        print("\n" + "=" * 60)
        t1 = time.perf_counter()
        new_toks = len(all_tokens)
    else:
        # Batch: generate all tokens then display
        logger.info("Generating response...")
        result = sample(model, params, inputs, sampling_cfg, rng, tokenizer=tokenizer)
        t1 = time.perf_counter()

        # Extract and display
        try:
            new_ids = result.tokens[0].tolist()
            full_ids = prompt_tokens + new_ids
            full_text = tokenizer.decode(full_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        except Exception:
            full_text = result.texts[0] if result.texts else ""

        assistant_response = extract_assistant(full_text)
        if not assistant_response:
            assistant_response = (result.texts[0] if result.texts else "").strip()

        print("\n" + "=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(assistant_response)
        print("=" * 60)

        try:
            new_toks = int(result.tokens.shape[1])
        except Exception:
            new_toks = 0

    elapsed = max(t1 - t0, 1e-6)
    tok_s = new_toks / elapsed if new_toks else 0.0
    print(f"Throughput: {tok_s:.2f} tok/s ({new_toks} tokens in {elapsed:.2f}s)")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
