# Qwen3‑VL written in JAX 🏄
<img width="1045" height="361" alt="Screenshot 2025-10-28 at 1 57 42 AM" src="https://github.com/user-attachments/assets/35734b42-6347-4bf1-b090-817ad5781244" />

A minimal, readable implementation of Qwen3‑VL inference in JAX/Flax(no PyTorch or HuggingFace(except tokenizers)!)

- `model.py` — Text decoder, vision encoder, mRoPE, GQA, loaders
- `sample.py` — Image preprocessing, prompting helpers, top‑k/top‑p sampling
- `utils.py` — Config (chz), logging, checkpoints, HF→JAX conversion
- `run.py` — Minimal CLI example

On a H200 I reach a whopping 0.5 tok/s because of JAX JIT issues I was running into that simply make it slower to run, despite the implementation being clean, this shouldn't be run in production.
  
## Quickstart

- Clone and convert HuggingFace weights to JAX
  
Modal:
```bash
 modal run modal_app.py --download
```
Local:
  ```bash
  git clone https://github.com/sdan/Qwen3-VL-JAX.git && cd Qwen3-VL-JAX
  uv sync  # CPU/default (use `uv sync --extra cuda12` for CUDA 12)

  # Download and convert weights (2B default)
  uv run huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir checkpoints/qwen3vl_2b
  uv run python -c "from utils import convert_hf_to_jax; convert_hf_to_jax('qwen3vl','./checkpoints/qwen3vl_2b')"
  ```

- Run inference (CLI)
 
Modal:
```
 modal run modal_app.py --image cat --prompt "Describe this cat"
```
Local:
  ```bash
  # Basic (default prompt: "What is shown in this image?")
  uv run python run.py inference.image=examples/imgs/horses.png

  # Custom prompt
  uv run python run.py inference.image=examples/imgs/horses.png inference.prompt="Describe this"

  # Streaming (tokens appear as generated)
  uv run python run.py inference.image=examples/imgs/horses.png inference.stream=true

  # Sampling params
  uv run python run.py inference.image=examples/imgs/horses.png sampling.temperature=0.8 sampling.max_new_tokens=256

  # CUDA
  JAX_PLATFORMS=gpu,cpu uv run python run.py inference.image=examples/imgs/horses.png inference.device=cuda
  ```

I saw ThinkingMachines use chz so I decided to make it first-class. This also allows you to easily swap config right in the CLI as such:
`uv run python run.py --image img.jpg sampling.temperature=0.95 sampling.max_new_tokens=512 model.dtype=float32`

Most of this code was taken from [sdan/vlm-gym](https://github.com/sdan/vlm-gym) as an attempt to cleanly abstract it out to sample from the policy optimization loop.
