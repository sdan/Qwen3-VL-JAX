#!/usr/bin/env python3
"""Qwen3-VL inference on Apple Silicon using MLX

Usage:
    python run_mlx.py examples/imgs/horses.png
    python run_mlx.py examples/imgs/horses.png "Describe this image in detail"
    python run_mlx.py examples/imgs/horses.png --max-tokens 200
"""
import argparse
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL with MLX on Apple Silicon")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("prompt", nargs="?", default="What is in this image?",
                        help="Prompt (default: 'What is in this image?')")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens to generate")
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct", help="Model name")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature")
    args = parser.parse_args()

    # Import MLX (lazy to show loading message first)
    print(f"Loading {args.model}...")
    t0 = time.perf_counter()

    from mlx_vlm import load, stream_generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config
    import mlx.core as mx

    model, processor = load(args.model)
    config = load_config(args.model)
    load_time = time.perf_counter() - t0

    print(f"Loaded in {load_time:.1f}s | Device: {mx.default_device()}")
    print(f"Image: {args.image}")
    print(f"Prompt: {args.prompt}")
    print("-" * 50)

    # Format prompt
    formatted = apply_chat_template(processor, config, args.prompt, num_images=1)

    # Stream generation
    token_count = 0
    t_start = time.perf_counter()
    t_first = None

    for token in stream_generate(
        model, processor, formatted, args.image,
        max_tokens=args.max_tokens,
        temp=args.temp,
    ):
        if t_first is None:
            t_first = time.perf_counter()
        print(token.text, end="", flush=True)
        token_count += 1

    t_end = time.perf_counter()
    print("\n" + "-" * 50)

    # Stats
    prefill_time = t_first - t_start if t_first else 0
    decode_time = t_end - t_first if t_first else t_end - t_start
    tok_per_sec = token_count / decode_time if decode_time > 0 else 0

    print(f"Tokens: {token_count} | Prefill: {prefill_time:.2f}s | Decode: {tok_per_sec:.1f} tok/s")

if __name__ == "__main__":
    main()
