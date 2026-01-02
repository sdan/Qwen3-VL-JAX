"""End-to-end vision stack smoke test on real images (no internet).

Runs a simplified Qwen3-VL-ish vision encoder stack:
  shuffle -> (norm + attention + mlp) * depth -> merger -> unshuffle
and compares fused vs unfused attention outputs + timing.
"""

from __future__ import annotations

import math
from pathlib import Path

import modal

app = modal.App("accel")

script_dir = Path(__file__).parent.resolve()
local_dir = script_dir
image_dir = script_dir.parent / "examples" / "imgs"

base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch>=2.4.0", "triton>=3.0.0", "pillow", "numpy")
    .add_local_dir(local_dir, remote_path="/root/kernels")
    .add_local_dir(image_dir, remote_path="/root/images")
)


@app.function(image=base_image, gpu="H100", timeout=600)
def run_v16():
    import sys
    sys.path.insert(0, "/root/kernels")

    import os
    import torch
    import torch.nn as nn
    import triton
    import numpy as np
    from PIL import Image

    from vision_attention import TritonVisionAttention, UnfusedVisionAttention, compute_vision_rope_2d
    from window_attn import get_window_index, window_shuffle

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_size = 1152
    num_heads = 16
    head_dim = 72
    depth = 27
    intermediate_size = 4304

    patch_size = 14
    spatial_merge_size = 2
    temporal_patch_size = 2
    window_size = 112

    merge_unit = spatial_merge_size**2

    IMAGE_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    IMAGE_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def smart_resize(
        height: int,
        width: int,
        factor: int,
        min_pixels: int = 56 * 56,
        max_pixels: int = 12845056,
    ) -> tuple[int, int]:
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

    def image_to_grid_thw(path: str) -> tuple[int, list[tuple[int, int, int]]]:
        pil_img = Image.open(path).convert("RGB")
        width, height = pil_img.size

        factor = patch_size * spatial_merge_size
        new_h, new_w = smart_resize(height, width, factor)

        if (new_w, new_h) != (width, height):
            pil_img = pil_img.resize((new_w, new_h), Image.Resampling.BICUBIC)

        image_np = np.asarray(pil_img, dtype=np.float32) / 255.0
        image_np = (image_np - IMAGE_MEAN) / IMAGE_STD
        image_np = np.transpose(image_np, (2, 0, 1))[None, ...]  # (1,C,H,W)

        frames = image_np.shape[0]
        if temporal_patch_size > 1 and frames % temporal_patch_size != 0:
            pad = temporal_patch_size - (frames % temporal_patch_size)
            image_np = np.concatenate(
                [image_np, np.repeat(image_np[-1:], pad, axis=0)], axis=0
            )

        frames, _c, new_h, new_w = image_np.shape
        grid_t = frames // temporal_patch_size
        grid_h = new_h // patch_size
        grid_w = new_w // patch_size
        total_tokens = grid_t * grid_h * grid_w
        return int(total_tokens), [(int(grid_t), int(grid_h), int(grid_w))]

    fullatt_block_indexes = tuple(range(0, depth, 4))
    fullatt_set = set(int(i) for i in fullatt_block_indexes)

    class VisionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True, dtype=dtype)
            self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True, dtype=dtype)

        def forward(self, x):
            x = self.fc1(x)
            x = torch.nn.functional.gelu(x, approximate="tanh")
            return self.fc2(x)

    class VisionBlock(nn.Module):
        def __init__(self, attention_cls):
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.attn = attention_cls(
                hidden_size,
                num_heads,
                head_dim,
                spatial_merge_size=spatial_merge_size,
            )
            self.mlp = VisionMLP()

        def forward(self, x, cu_seqlens, cos, sin):
            x = x + self.attn(self.norm1(x), cu_seqlens=cu_seqlens, cos=cos, sin=sin)
            x = x + self.mlp(self.norm2(x))
            return x

    class PatchMerger(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_size, eps=1e-6, dtype=dtype)
            self.fc1 = nn.Linear(
                hidden_size * merge_unit, hidden_size * merge_unit, bias=True, dtype=dtype
            )
            self.fc2 = nn.Linear(hidden_size * merge_unit, hidden_size, bias=True, dtype=dtype)

        def forward(self, x):
            x = self.norm(x)
            x = x.reshape(-1, hidden_size * merge_unit)
            x = torch.nn.functional.gelu(self.fc1(x), approximate="tanh")
            return self.fc2(x)

    class VisionEncoder(nn.Module):
        def __init__(self, attention_cls):
            super().__init__()
            self.blocks = nn.ModuleList([VisionBlock(attention_cls) for _ in range(depth)])
            self.merger = PatchMerger()

        def forward(self, x, cu_full, cu_window, cos, sin, *, window_idx):
            use_window = cu_window is not None and window_idx is not None and len(fullatt_set) < depth
            for i, blk in enumerate(self.blocks):
                cu = cu_full if (not use_window or i in fullatt_set) else cu_window
                x = blk(x, cu, cos, sin)
            x = self.merger(x)
            if use_window:
                reverse_idx = torch.argsort(window_idx)
                x = x[reverse_idx, :]
            return x

    fused = VisionEncoder(TritonVisionAttention).to(device, dtype)
    unfused = VisionEncoder(UnfusedVisionAttention).to(device, dtype)
    unfused.load_state_dict(fused.state_dict(), strict=True)

    print("=" * 70)
    print("VISION STACK (real images): fused vs unfused")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}  dtype={dtype}")
    print(
        f"hidden={hidden_size} heads={num_heads} head_dim={head_dim} depth={depth} "
        f"window_size={window_size} patch={patch_size} merge={spatial_merge_size} fullatt={sorted(fullatt_set)}"
    )

    image_dir_path = "/root/images"
    test_images = sorted(
        f for f in os.listdir(image_dir_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"\nFound {len(test_images)} images")

    results = []
    for img_name in test_images:
        img_path = os.path.join(image_dir_path, img_name)
        total_tokens, grid_thw = image_to_grid_thw(img_path)
        assert total_tokens % merge_unit == 0

        # Window shuffle config + full cu_seqlens.
        window_idx, cu_window = get_window_index(
            grid_thw,
            window_size=window_size,
            spatial_merge_size=spatial_merge_size,
            patch_size=patch_size,
            device=device,
        )
        assert cu_window[0].item() == 0 and cu_window[-1].item() == total_tokens

        token_counts = [int(t) * int(h) * int(w) for (t, h, w) in grid_thw]
        cu_full = torch.tensor(
            [0] + list(torch.tensor(token_counts, device=device).cumsum(0).tolist()),
            device=device,
            dtype=torch.int32,
        )
        assert cu_full[-1].item() == total_tokens

        cos, sin = compute_vision_rope_2d(
            grid_thw,
            head_dim=head_dim,
            spatial_merge_size=spatial_merge_size,
            rope_theta=10000.0,
            device=device,
            dtype=dtype,
            window_idx=window_idx,
        )

        x = torch.randn(total_tokens, hidden_size, device=device, dtype=dtype)
        x = window_shuffle(x, window_idx, spatial_merge_size=spatial_merge_size)

        # Compile/autotune once per shape (excluded from timing).
        _ = fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
        _ = unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
        torch.cuda.synchronize()

        out_fused = fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
        out_unfused = unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx)
        torch.cuda.synchronize()

        diff = (out_fused - out_unfused).float().abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        has_nan = torch.isnan(out_fused).any().item()
        has_inf = torch.isinf(out_fused).any().item()

        fused_ms = triton.testing.do_bench(
            lambda: fused(x, cu_full, cu_window, cos, sin, window_idx=window_idx),
            warmup=5,
            rep=15,
        )
        unfused_ms = triton.testing.do_bench(
            lambda: unfused(x, cu_full, cu_window, cos, sin, window_idx=window_idx),
            warmup=5,
            rep=15,
        )

        results.append(
            {
                "image": img_name,
                "tokens": int(total_tokens),
                "grid_thw": grid_thw[0],
                "fused_ms": float(fused_ms),
                "unfused_ms": float(unfused_ms),
                "speedup": float(unfused_ms / fused_ms),
                "max_abs": float(max_abs),
                "mean_abs": float(mean_abs),
                "nan": bool(has_nan),
                "inf": bool(has_inf),
            }
        )

        status = "✓" if (not has_nan and not has_inf and max_abs < 5e-2) else "✗"
        print(
            f"{img_name}: seq={total_tokens:<6} grid={grid_thw[0]} "
            f"max_abs={max_abs:.3e} fused={fused_ms:.3f}ms unfused={unfused_ms:.3f}ms "
            f"speedup={unfused_ms / fused_ms:.2f}x {status}"
        )

    return results


if __name__ == "__main__":
    with app.run():
        run_v16.remote()
