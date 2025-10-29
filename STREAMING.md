# Streaming Generation

This codebase now supports **real-time token-by-token streaming** for faster perceived inference.

## Quick Start

### 1. Streaming Generation (Real-time tokens)

```bash
uv run python run_streaming.py inference.image=examples/imgs/panda_climbing.png inference.prompt="Describe this image"
```

You'll see tokens appear in real-time as they're generated, ChatGPT-style!

### 2. Batch Generation (Original, faster total time)

```bash
uv run python run.py inference.image=examples/imgs/panda_climbing.png inference.prompt="Describe this image"
```

Waits for all tokens, then shows the complete response.

## More Examples

Try different images:

```bash
# Coffee shop scene
uv run python run_streaming.py inference.image=examples/imgs/coffee_laptop.png inference.prompt="What's in this photo?"

# Mountain landscape
uv run python run_streaming.py inference.image=examples/imgs/mountain_landscape.png inference.prompt="Describe the scenery"

# F-35 jet
uv run python run_streaming.py inference.image=examples/imgs/f35_takeoff.png inference.prompt="What aircraft is this?"
```

Adjust sampling parameters:

```bash
# More creative (higher temperature)
uv run python run_streaming.py inference.image=examples/imgs/panda_climbing.png \
  inference.prompt="Write a story about this image" \
  sampling.temperature=0.9 sampling.max_new_tokens=256

# More focused (lower temperature)
uv run python run_streaming.py inference.image=examples/imgs/stadium_aerial.png \
  inference.prompt="What do you see?" \
  sampling.temperature=0.3 sampling.top_p=0.8
```

## Usage

### Basic Streaming (Recommended)

```bash
uv run python run_streaming.py inference.image=examples/imgs/panda_climbing.png inference.prompt="Describe this image"
```

Tokens will appear in real-time as they're generated, similar to ChatGPT's interface.

### Programmatic API

```python
from sample import sample_streaming, SamplingConfig, VLMInputs
import jax

# ... load model, params, vision_embeddings, tokenizer ...

sampling_cfg = SamplingConfig(
    temperature=0.7,
    top_p=0.9,
    top_k=1024,
    max_new_tokens=512,
    eos_id=eos_id,
    pad_id=pad_id,
)

inputs = VLMInputs(
    prompt_tokens=jnp.array([prompt_tokens], dtype=jnp.int32),
    vision=vision_embeddings,
    grid_thw=grid_thw,
    image_pad_id=image_pad_id,
)

rng = jax.random.PRNGKey(42)

# Stream tokens one at a time
for token_id, text, logprob in sample_streaming(model, params, inputs,
                                                 sampling_cfg, rng,
                                                 tokenizer=tokenizer):
    print(text, end='', flush=True)
```

### Batch Generation (Original API)

For batch processing, use the original `sample()` function:

```python
from sample import sample

result = sample(model, params, inputs, sampling_cfg, rng, tokenizer=tokenizer)
print(result.texts[0])  # All tokens generated at once
```

## Performance

- **Streaming**: Uses Python loop with JIT-compiled single-step function
  - Pros: Real-time output, lower perceived latency
  - Cons: Slightly slower total throughput due to Python loop overhead
  - **Batch size**: Only supports batch_size=1

- **Batch**: Uses `jax.lax.scan` for fully compiled loop
  - Pros: Maximum throughput, supports batching
  - Cons: No intermediate output until complete
  - **Batch size**: Supports any batch size

## Hardware Acceleration

On Mac M-series, ensure you install JAX Metal for GPU acceleration:

```bash
uv pip install jax-metal
```

Check active backend:
```bash
python -c "import jax; print(jax.devices())"
```

Expected output:
- **CPU**: `[CpuDevice(id=0)]`
- **Metal (GPU)**: `[METAL(id=0)]`
