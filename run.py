#!/usr/bin/env python3
"""Simple Qwen3-VL inference script

Usage:
    python run.py --image path/to/image.jpg --prompt "What is in this image?"
    python run.py --image image.jpg model.model_dir=./my_checkpoint sampling.temperature=0.8
"""
from __future__ import annotations

import argparse

import jax, jax.numpy as jnp
import chz
from transformers import AutoTokenizer

from model import create_model_from_ckpt
from sample import (SamplingConfig, VLMInputs, sample, preprocess_image,
                   chat_prompt_with_image, extract_assistant)
from utils import Config, setup_logger


def main(cfg: Config) -> None:
    # Parse extra args
    parser = argparse.ArgumentParser(description="Qwen3-VL inference")
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--prompt", type=str, default="What is shown in this image?",
                       help="Text prompt")
    args, _ = parser.parse_known_args()

    # Setup logger
    logger = setup_logger(__name__, level=cfg.logging.log_level,
                         log_file=cfg.logging.log_file, verbose=cfg.logging.verbose)

    # Load model
    logger.info(f"Loading model from {cfg.model.model_dir}")
    model, params = create_model_from_ckpt(cfg.model.model_dir)
    logger.info(f"Model: {model.spec.text.num_hidden_layers} layers, "
               f"{model.spec.text.hidden_size} hidden dim")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_dir, trust_remote_code=True)
    logger.info(f"Tokenizer: vocab size {len(tokenizer)}")

    # Preprocess image
    logger.info(f"Processing image: {args.image}")
    pixel_values, grid_thw = preprocess_image(
        args.image,
        patch_size=model.spec.vision.patch_size,
        spatial_merge_size=model.spec.vision.spatial_merge_size,
        temporal_patch_size=model.spec.vision.temporal_patch_size,
    )
    logger.debug(f"Image shape: {pixel_values.shape}, grid: {grid_thw}")

    # Encode vision
    logger.info("Encoding vision features...")
    vision_embeddings = model.apply(
        {"params": params},
        pixel_values[None],
        grid_thw[None],
        method=model.encode_vision,
    )
    num_vision_tokens = vision_embeddings.tokens.shape[1]
    logger.info(f"Vision tokens: {num_vision_tokens}")

    # Format prompt
    formatted_prompt = chat_prompt_with_image(num_vision_tokens, args.prompt)
    prompt_tokens = tokenizer.encode(formatted_prompt, add_special_tokens=False)
    logger.debug(f"Prompt length: {len(prompt_tokens)} tokens")

    # Get special token IDs
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    eos_id = tokenizer.eos_token_id
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
        grid_thw=jnp.array([grid_thw], dtype=jnp.int32),
        image_pad_id=image_pad_id,
    )

    # Generate
    logger.info("Generating response...")
    rng = jax.random.PRNGKey(cfg.sampling.seed)
    result = sample(model, params, inputs, sampling_cfg, rng, tokenizer=tokenizer)

    # Extract and display
    response_text = result.texts[0] if result.texts else ""
    assistant_response = extract_assistant(response_text)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(assistant_response)
    print("=" * 60)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
