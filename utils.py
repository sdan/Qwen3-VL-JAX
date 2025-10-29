"""Utilities: config, checkpointing, logging, HF conversion

All non-model utilities in one file for simplicity.
"""
from __future__ import annotations

import concurrent.futures
import logging, os, pickle, shutil, sys, time
from typing import Optional

import chz
from flax.core import unfreeze
from huggingface_hub import snapshot_download

# ============================================================================
# Logging
# ============================================================================

def setup_logger(name: str = "qwen3vl", level: str = "INFO", log_file: Optional[str] = None,
                verbose: bool = False) -> logging.Logger:
    """Configure logger with optional file output"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()

    formatter = (logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                                  datefmt="%Y-%m-%d %H:%M:%S") if verbose
                else logging.Formatter("%(levelname)s: %(message)s"))

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    logger.propagate = False
    return logger


def get_logger(name: str = "qwen3vl") -> logging.Logger:
    """Get existing logger (or default if not configured)"""
    return logging.getLogger(name)


class LoggerContext:
    """Temporarily change log level

    Example:
        with LoggerContext(logger, "WARNING"):
            logger.info("Suppressed")  # Hidden
    """
    def __init__(self, logger: logging.Logger, level: str):
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)
        return False


# ============================================================================
# Checkpointing
# ============================================================================

def parent_dir(filename: str) -> str:
    return filename.rsplit('/', 1)[0]


def name(filename: str) -> str:
    return filename.rsplit('/', 1)[1]


class Checkpoint:
    """Simple checkpoint saver/loader compatible with gs:// buckets

    From https://github.com/danijar/elements/blob/main/elements/checkpoint.py
    """
    def __init__(self, filename: str, parallel: bool = True):
        self._filename = filename
        self._values = {}
        self._parallel = parallel
        if self._parallel:
            self._worker = concurrent.futures.ThreadPoolExecutor(1, 'checkpoint')
            self._promise = None

    def __setattr__(self, name: str, value):
        if name in ('exists', 'save', 'load'):
            return super().__setattr__(name, value)
        if name.startswith('_'):
            return super().__setattr__(name, value)
        self._values[name] = value

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)
        try:
            return self._values[name]
        except KeyError:
            raise ValueError(name)

    def set_model(self, model):
        """Store all savable attributes from model"""
        for key in model.__dict__.keys():
            data = getattr(model, key)
            if hasattr(data, 'save') or key == 'config':
                self._values[key] = data

    def save(self, filename: Optional[str] = None, keys: Optional[list] = None):
        """Save checkpoint"""
        filename = filename or self._filename
        logger = get_logger(__name__)
        logger.info(f'Writing checkpoint: {filename}')
        if self._parallel:
            if self._promise:
                self._promise.result()
            self._promise = self._worker.submit(self._save, filename, keys)
        else:
            self._save(filename, keys)

    def _save(self, filename: str, keys: Optional[list]):
        keys = tuple(self._values.keys() if keys is None else keys)
        data = {k: (self._values[k].save() if hasattr(self._values[k], 'save') else self._values[k])
               for k in keys}
        data['_timestamp'] = time.time()
        content = pickle.dumps(data)

        if 'gs://' in filename:
            import tensorflow as tf
            tf.io.gfile.makedirs(parent_dir(filename))
            with tf.io.gfile.GFile(filename, 'wb') as f:
                f.write(content)
        else:
            os.makedirs(parent_dir(filename), exist_ok=True)
            tmp = parent_dir(filename) + '/' + name(filename) + '.tmp'
            with open(tmp, 'wb') as f:
                f.write(content)
            shutil.move(tmp, filename)
        get_logger(__name__).info('Wrote checkpoint.')

    def load_as_dict(self, filename: Optional[str] = None) -> dict:
        """Load checkpoint as dictionary"""
        filename = filename or self._filename
        if 'gs://' in filename:
            import tensorflow as tf
            with tf.io.gfile.GFile(filename, 'rb') as f:
                data = pickle.loads(f.read())
        else:
            with open(filename, 'rb') as f:
                data = pickle.loads(f.read())

        age = time.time() - data['_timestamp']
        get_logger(__name__).info(f'Loaded checkpoint from {age:.0f} seconds ago.')
        return data

    def load_model(self, model, filename: Optional[str] = None):
        """Load checkpoint into model"""
        cp_dict = self.load_as_dict(filename)
        replace_dict = {}
        for key in model.__dict__.keys():
            if key in cp_dict and key != 'config':
                replace_dict[key] = getattr(model, key).load(cp_dict[key])
        return model.replace(**replace_dict)


# ============================================================================
# Configuration
# ============================================================================

@chz.chz
class ModelConfig:
    """Model loading configuration"""
    model_dir: str = "checkpoints/qwen3vl_2b"
    """Path to checkpoint directory"""
    dtype: str = "bfloat16"
    """Model dtype: bfloat16, float32, or float16"""
    skip_deepstack: bool = False
    """Skip deepstack vision features for faster inference"""


@chz.chz
class SamplingConfig:
    """Sampling hyperparameters"""
    temperature: float = 0.7
    """Sampling temperature (higher = more random)"""
    top_p: float = 0.9
    """Nucleus sampling threshold"""
    top_k: int = 1024
    """Top-k vocabulary shortlist"""
    max_new_tokens: int = 512
    """Maximum tokens to generate"""
    seed: int = 42
    """Random seed"""


@chz.chz
class TrainingConfig:
    """Training hyperparameters (for future fine-tuning)"""
    learning_rate: float = 1e-6
    """AdamW learning rate"""
    weight_decay: float = 1e-2
    """Weight decay"""
    batch_size: int = 8
    """Batch size"""
    num_steps: int = 1000
    """Total steps"""
    max_grad_norm: float = 1.0
    """Gradient clipping"""
    warmup_steps: int = 100
    """LR warmup steps"""
    save_every: int = 100
    """Checkpoint frequency"""


@chz.chz
class LoggingConfig:
    """Logging configuration"""
    log_level: str = "INFO"
    """Logging level"""
    log_file: str | None = None
    """Optional log file"""
    verbose: bool = False
    """Verbose output with timestamps"""


@chz.chz
class InferenceConfig:
    """Inference-specific configuration"""
    image: str | None = None
    """Path to image file"""
    prompt: str = "What is shown in this image?"
    """Text prompt for the model"""
    device: str = "auto"
    """Device preference: auto|cpu|cuda(gpu)"""


@chz.chz
class Config:
    """Root configuration container

    Usage:
        def main(cfg: Config):
            print(cfg.model.model_dir)

        if __name__ == "__main__":
            chz.nested_entrypoint(main)

    CLI overrides:
        python script.py model.model_dir=./my_model sampling.temperature=0.8
    """
    model: ModelConfig
    sampling: SamplingConfig
    training: TrainingConfig
    logging: LoggingConfig
    inference: InferenceConfig


def run_with_config(main_fn):
    """Decorator for config-based entry point

    Example:
        @run_with_config
        def main(cfg: Config):
            # Your code
            pass
    """
    return chz.nested_entrypoint(main_fn)


# ============================================================================
# HuggingFace conversion
# ============================================================================

def _ensure_trailing_slash(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _resolve_hf_dir(hf_dir: Optional[str], hf_repo: str, revision: Optional[str] = None,
                   cache_dir: Optional[str] = None) -> str:
    """Return local directory with HF weights (download if needed)"""
    if hf_dir:
        return _ensure_trailing_slash(os.path.expanduser(hf_dir))

    snapshot_path = snapshot_download(
        repo_id=hf_repo,
        revision=revision,
        cache_dir=None if cache_dir is None else os.path.expanduser(cache_dir),
        local_files_only=False,
    )
    return _ensure_trailing_slash(snapshot_path)


def convert_hf_to_jax(model_type: str, model_dir: str, hf_dir: Optional[str] = None,
                     hf_repo: Optional[str] = None, revision: Optional[str] = None,
                     cache_dir: Optional[str] = None):
    """Convert HuggingFace checkpoint to JAX format

    Args:
        model_type: "qwen3vl" or "qwen25vl"
        model_dir: Output directory for converted checkpoint
        hf_dir: Optional local HF directory (if omitted, downloads from hf_repo)
        hf_repo: HF repo ID (uses default if omitted)
        revision: Optional HF revision
        cache_dir: Optional HF cache directory
    """
    logger = setup_logger(__name__, level="INFO")

    # Default repos
    if hf_repo is None:
        hf_repo = {
            "qwen25vl": "Qwen/Qwen2.5-VL-7B-Instruct",
            "qwen3vl": "Qwen/Qwen3-VL-2B-Instruct",
        }[model_type]

    ckpt_dir = _ensure_trailing_slash(os.path.expanduser(model_dir))
    os.makedirs(ckpt_dir, exist_ok=True)

    # Download/locate HF weights
    hf_dir = _resolve_hf_dir(hf_dir, hf_repo, revision, cache_dir)
    logger.info(f"Using HF snapshot: {hf_dir}")

    # Convert
    import model as model_module
    _, params = model_module.create_model_from_hf(hf_dir)
    params = unfreeze(params)

    # Save
    ckpt = Checkpoint(os.path.join(ckpt_dir, "params.pkl"), parallel=False)
    ckpt.params = params
    ckpt.save()

    # Copy config files
    base_files = [
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "preprocessor_config.json",
        "generation_config.json",
        "chat_template.json",
    ]
    if model_type == "qwen3vl":
        base_files.append("video_preprocessor_config.json")

    for fname in base_files:
        src = os.path.join(hf_dir, fname)
        dst = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Copied: {fname}")
        else:
            logger.warning(f"{fname} not found, skipped")

    logger.info(f"Conversion complete: {ckpt_dir}")


__all__ = [
    "setup_logger", "get_logger", "LoggerContext",
    "Checkpoint",
    "Config", "ModelConfig", "SamplingConfig", "TrainingConfig", "LoggingConfig", "InferenceConfig",
    "run_with_config", "convert_hf_to_jax",
]
