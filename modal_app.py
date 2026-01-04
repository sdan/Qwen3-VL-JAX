"""Modal stub for Qwen3-VL inference

Usage:
    # Deploy and run
    modal run modal_app.py --image-url "https://example.com/image.jpg"
    modal run modal_app.py --image-url "https://example.com/image.jpg" --prompt "Describe this"

    # Deploy as endpoint
    modal deploy modal_app.py
    # Then POST to the endpoint with {"image_url": "...", "prompt": "..."}
"""
import modal

app = modal.App("qwen3-vl-jax")

# Build image with JAX + CUDA and mount local Python modules
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "jax[cuda12]",
        "flax",
        "transformers>=4.48.0",  # Recent version with compatible huggingface-hub
        "safetensors",
        "huggingface_hub>=0.34.0,<1.0",  # Explicit compatible version
        "chz",
        "pillow",
        "requests",
    )
    # Mount local Python files (Modal 1.0+ API)
    .add_local_file("model.py", "/root/model.py")
    .add_local_file("sample.py", "/root/sample.py")
    .add_local_file("utils.py", "/root/utils.py")
)

# Volume for caching model weights
volume = modal.Volume.from_name("qwen3-vl-weights", create_if_missing=True)
MODEL_DIR = "/weights/qwen3vl_2b"


@app.function(
    image=image,
    gpu="H200",
    volumes={"/weights": volume},
    timeout=600,
)
def download_model():
    """Download and convert model weights (run once)"""
    import os
    if os.path.exists(f"{MODEL_DIR}/params.pkl"):
        print("Model already downloaded")
        return

    from huggingface_hub import snapshot_download

    # Download HF weights
    hf_dir = snapshot_download(
        "Qwen/Qwen3-VL-2B-Instruct",
        local_dir="/weights/hf_qwen3vl_2b",
    )
    print(f"Downloaded to {hf_dir}")

    # Convert to JAX
    import sys
    sys.path.insert(0, "/root")

    from model import create_model_from_hf
    from flax.core import unfreeze
    import pickle
    import shutil

    print("Converting to JAX...")
    _, params = create_model_from_hf(hf_dir)
    params = unfreeze(params)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/params.pkl", "wb") as f:
        pickle.dump({"params": params}, f)

    # Copy config files
    for fname in ["config.json", "tokenizer_config.json", "tokenizer.json"]:
        src = os.path.join(hf_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, f"{MODEL_DIR}/{fname}")

    volume.commit()
    print("Done!")


@app.cls(
    image=image,
    gpu="H100",
    volumes={"/weights": volume},
    timeout=900,  # 15 min for first-run JIT compilation
)
class Qwen3VL:
    @modal.enter()
    def load_model(self):
        import os
        # Disable XLA autotuning to speed up first compilation (use default kernels)
        os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

        import jax
        print(f"JAX devices: {jax.devices()}")

        from model import create_model_from_ckpt
        from transformers import AutoTokenizer

        self.model, self.params = create_model_from_ckpt(MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        print(f"Model loaded: {self.model.spec.text.num_hidden_layers} layers")

    @modal.method()
    def inference(self, image_url: str, prompt: str = "What is shown in this image?",
                  temperature: float = 0.7, max_tokens: int = 32) -> str:  # Reduced for faster testing
        import jax
        import jax.numpy as jnp
        import requests
        from io import BytesIO
        from PIL import Image

        from sample import (
            SamplingConfig, VLMInputs, sample,
            preprocess_image, chat_prompt_with_image, extract_assistant
        )

        # Download image
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Qwen3VL/1.0)"}
        response = requests.get(image_url, headers=headers, timeout=30)
        response.raise_for_status()
        pil_image = Image.open(BytesIO(response.content)).convert("RGB")

        # Preprocess
        pixel_values, grid_thw = preprocess_image(
            pil_image,
            patch_size=self.model.spec.vision.patch_size,
            spatial_merge_size=self.model.spec.vision.spatial_merge_size,
            temporal_patch_size=self.model.spec.vision.temporal_patch_size,
        )

        # Encode vision
        vision_emb = self.model.apply(
            {"params": self.params},
            pixel_values,
            grid_thw,
            method=self.model.encode_vision,
        )
        num_vision_tokens = int(vision_emb.tokens.shape[0] if vision_emb.tokens.ndim == 2
                                else vision_emb.tokens.shape[1])

        # Format prompt
        formatted = chat_prompt_with_image(num_vision_tokens, prompt)
        prompt_tokens = self.tokenizer.encode(formatted, add_special_tokens=False)

        # Get special tokens
        image_pad_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        vision_start_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        eos_id = im_end_id if im_end_id and int(im_end_id) >= 0 else self.tokenizer.eos_token_id

        # Sample
        cfg = SamplingConfig(
            temperature=temperature,
            top_p=0.9,
            top_k=1024,
            max_new_tokens=max_tokens,
            eos_id=eos_id,
            pad_id=0,
        )

        inputs = VLMInputs(
            prompt_tokens=jnp.array([prompt_tokens], dtype=jnp.int32),
            vision=vision_emb,
            grid_thw=grid_thw,
            image_pad_id=image_pad_id,
            vision_start_id=vision_start_id,
        )

        result = sample(
            self.model, self.params, inputs, cfg,
            jax.random.PRNGKey(42), tokenizer=self.tokenizer
        )

        # Extract response
        new_ids = result.tokens[0].tolist()
        full_ids = prompt_tokens + new_ids
        full_text = self.tokenizer.decode(full_ids, skip_special_tokens=False)
        response = extract_assistant(full_text) or result.texts[0]

        return response.strip()


# Example images (URLs for remote inference) - using reliable free image services
EXAMPLE_IMAGES = {
    "cat": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=800",  # Orange cat
    "dog": "https://images.unsplash.com/photo-1587300003388-59208cc962cb?w=800",  # Golden retriever
    "food": "https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=800",  # Healthy food bowl
    "city": "https://images.unsplash.com/photo-1480714378408-67cf0d13bc1b?w=800",  # NYC skyline
    "car": "https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=800",  # Classic car
    "nature": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?w=800",  # Mountain landscape
}


@app.local_entrypoint()
def main(
    image_url: str = "",
    image: str = "cat",  # Use example image name: cat, dog, food, city, car
    prompt: str = "What is shown in this image?",
    download: bool = False,
):
    """Run Qwen3-VL inference on Modal.

    Examples:
        modal run modal_app.py --download              # First time: download weights
        modal run modal_app.py                         # Default: cat image
        modal run modal_app.py --image dog             # Use example image
        modal run modal_app.py --image-url "https://..."  # Custom URL
        modal run modal_app.py --image cat --prompt "Count the whiskers"
    """
    if download:
        download_model.remote()
        return

    # Resolve image URL
    if image_url:
        url = image_url
    elif image in EXAMPLE_IMAGES:
        url = EXAMPLE_IMAGES[image]
        print(f"Using example image: {image}")
    else:
        print(f"Unknown image '{image}'. Available: {list(EXAMPLE_IMAGES.keys())}")
        return

    model = Qwen3VL()
    result = model.inference.remote(image_url=url, prompt=prompt)
    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(result)
    print("=" * 60)
