# Qwen3‑VL (JAX)

A minimal, readable implementation of Qwen3‑VL inference in JAX/Flax.

The code is compact and focused on clarity. No PyTorch dependency.

## Quickstart

- Clone and convert HuggingFace weights to JAX
  - `git clone https://github.com/sdan/Qwen3-VL-JAX.git && cd Qwen3-VL-JAX`
  - `huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir checkpoints/qwen3vl_2b`
  - `uv sync`
  - `uv run python -c "from utils import convert_hf_to_jax; convert_hf_to_jax('qwen3vl','./checkpoints/qwen3vl_2b')"`

- Run inference (CLI)
  - `uv run python run.py --image examples/imgs/demo.jpg --prompt "What is in this image?"`
  - With overrides: `uv run python run.py --image examples/imgs/demo.jpg --prompt "Describe this" model.model_dir=./checkpoints/qwen3vl_2b sampling.temperature=0.8 sampling.max_new_tokens=256`

## Minimal Python Example

```python
import jax
from model import create_model_from_ckpt
from sample import sample, SamplingConfig, VLMInputs, preprocess_image, chat_prompt_with_image
from transformers import AutoTokenizer

model, params = create_model_from_ckpt("./checkpoints/qwen3vl_2b")
tokenizer = AutoTokenizer.from_pretrained("./checkpoints/qwen3vl_2b", trust_remote_code=True)

pixel_values, grid_thw = preprocess_image("image.jpg",
    patch_size=model.spec.vision.patch_size,
    spatial_merge_size=model.spec.vision.spatial_merge_size,
    temporal_patch_size=model.spec.vision.temporal_patch_size)

vision_emb = model.apply({"params": params}, pixel_values[None], grid_thw[None], method=model.encode_vision)
num_vision_tokens = vision_emb.tokens.shape[1]
prompt = chat_prompt_with_image(num_vision_tokens, "What is in this image?")
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

inputs = VLMInputs(
    prompt_tokens=jax.numpy.array([prompt_tokens], dtype=jax.numpy.int32),
    vision=vision_emb,
    grid_thw=grid_thw,
    image_pad_id=tokenizer.convert_tokens_to_ids("<|image_pad|>"))

cfg = SamplingConfig(temperature=0.7, top_p=0.9, top_k=1024, max_new_tokens=256,
                     eos_id=tokenizer.eos_token_id, pad_id=0)
result = sample(model, params, inputs, cfg, jax.random.PRNGKey(42), tokenizer=tokenizer)
print(result.texts[0])
```

## Files

- `model.py` — Text decoder, vision encoder, mRoPE, GQA, loaders
- `sample.py` — Image preprocessing, prompting helpers, top‑k/top‑p sampling
- `utils.py` — Config (chz), logging, checkpoints, HF→JAX conversion
- `run.py` — Minimal CLI example

## Model At A Glance

- Decoder: 28 layers, 2048 hidden, GQA (16 Q heads, 8 KV heads)
- Vision: ViT with window attention and 2×2 spatial merge
- Positional encoding: 1D RoPE for text, 3D mRoPE for vision (t/h/w)
- Tricks: QK‑norm, grouped‑query attention (2× smaller KV cache)

## Config & Overrides

- Type‑checked configs with [chz](https://github.com/openai/chz)
- Override any field from the CLI, e.g.
  - `uv run python run.py --image img.jpg sampling.temperature=0.95 sampling.max_new_tokens=512 model.dtype=float32`

## Performance (2B, bf16, Apple M1 Max)

- Prefill ~150 ms (512 text + 1k vision)
- Decode ~30 ms/token (~33 tok/s)
- ~6 GB total memory (weights+cache+acts)

## Philosophy

- Minimal, flat, readable; delete before adding
- JAX‑first: jit, scan, PyTrees; no hidden state
- One place per idea; no deep indirection

## Why uv

- Fast, deterministic, and built‑in virtualenvs
- `uv sync` to install, `uv run python run.py` to execute

## Testing

- Imports: `uv run python -c "import model, sample, utils"`
- Demo: `uv run python run.py --image examples/imgs/demo.jpg --prompt "Describe this image"`

## Credits

- Qwen team (model)
- Checkpoint helpers adapted from community projects listed in code

## License

Apache‑2.0
