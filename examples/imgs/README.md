# Example Images - Memory Optimized

All images in this directory have been resized to **256×256 pixels** for optimal memory efficiency.

## Vision Token Count

Each 256×256 image generates exactly **256 vision tokens** when processed by Qwen3-VL:

- Grid dimensions: 16×16 (256 ÷ patch_size=16)
- Vision tokens: 1 × 16 × 16 = **256 tokens**
- Memory per image: ~1 MB (4 KB/token after spatial merging)

## Optimization Results

**Before resizing:**
- Original sizes: 800×531 to 2880×1922
- Vision tokens: 1,700 to 21,600 per image
- Total tokens: 54,432 across 8 images
- File size: ~15.5 MB total

**After resizing:**
- All images: 256×256
- Vision tokens: 256 per image consistently
- Total tokens: 2,048 across 8 images (96.2% reduction)
- File size: ~746 KB total (95.2% reduction)

## Image Processing

Images are processed with:
- `patch_size`: 16
- `spatial_merge_size`: 2
- `temporal_patch_size`: 2
- High-quality LANCZOS resampling

## Use Cases

These optimized images are ideal for:
- Low-memory inference on edge devices
- Fast prototyping and testing
- Demonstrations with minimal VRAM requirements
- Mobile and embedded deployments

For higher quality inference, you can use larger images (up to 12.8MP max), but expect proportionally more vision tokens and memory usage.
