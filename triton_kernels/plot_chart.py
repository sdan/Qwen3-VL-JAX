"""Generate the one chart as PNG.

Run after bench_chart.py to use real data:
    python3 plot_chart.py

Or run standalone with sample data:
    python3 plot_chart.py --sample
"""

import argparse
from pathlib import Path

# Sample data is intentionally synthetic (do not treat it as a performance claim).
SAMPLE_DATA = [
    {"img_size": 448, "flash_rope_ms": 1.00, "fused_ms": 0.50, "speedup": 2.0},
    {"img_size": 672, "flash_rope_ms": 1.10, "fused_ms": 0.55, "speedup": 2.0},
    {"img_size": 896, "flash_rope_ms": 1.25, "fused_ms": 0.62, "speedup": 2.0},
    {"img_size": 1344, "flash_rope_ms": 1.60, "fused_ms": 0.80, "speedup": 2.0},
]


def load_data(csv_path: Path) -> list[dict]:
    """Load data from CSV or use sample."""
    if csv_path.exists():
        data = []
        lines = csv_path.read_text().strip().split("\n")[1:]  # Skip header
        for line in lines:
            parts = line.split(",")
            data.append({
                "img_size": int(parts[0]),
                "flash_rope_ms": float(parts[2]),
                "fused_ms": float(parts[3]),
                "speedup": float(parts[4]),
            })
        return data
    raise FileNotFoundError(f"CSV not found: {csv_path}")


def create_chart(data: list[dict], output_path: Path):
    """Create the bar chart."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    # Style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Data
    labels = [f"{d['img_size']}²" for d in data]
    flash_times = [d["flash_rope_ms"] for d in data]
    fused_times = [d["fused_ms"] for d in data]
    speedups = [d["speedup"] for d in data]

    x = range(len(labels))
    width = 0.35

    # Bars
    bars1 = ax.bar([i - width/2 for i in x], flash_times, width,
                   label='Flash + RoPE', color='#ff6b6b', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar([i + width/2 for i in x], fused_times, width,
                   label='Fused (Ours)', color='#4ecdc4', edgecolor='black', linewidth=0.5)

    # Speedup labels
    for i, (bar, speedup) in enumerate(zip(bars1, speedups)):
        ax.annotate(f'{speedup:.1f}×',
                    xy=(i, max(flash_times[i], fused_times[i]) + 0.02),
                    ha='center', va='bottom',
                    fontsize=12, fontweight='bold', color='#2d3436')

    # Labels and title
    ax.set_xlabel('Image Size (pixels)', fontsize=12)
    ax.set_ylabel('Time (ms, per attention layer)', fontsize=12)
    ax.set_title('Fused RoPE+Attention vs Flash-Attention-2 + Separate RoPE\n'
                 'Qwen3-VL Vision Encoder (attention only) on H100', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='upper left')

    # Grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # Add insight at bottom
    avg_speedup = sum(speedups) / len(speedups)
    fig.text(0.5, 0.02,
             f'Average: {avg_speedup:.1f}× faster by eliminating memory round-trip',
             ha='center', fontsize=11, style='italic', color='#636e72')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")

    # Also save SVG for blog
    svg_path = output_path.with_suffix('.svg')
    plt.savefig(svg_path, bbox_inches='tight', facecolor='white')
    print(f"Saved: {svg_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    csv_path = script_dir / "benchmark_results.csv"
    output_path = script_dir / "benchmark_chart.png"

    if args.sample:
        data = SAMPLE_DATA
        print("Using sample data")
    else:
        try:
            data = load_data(csv_path)
        except FileNotFoundError:
            print(f"CSV not found: {csv_path}")
            print("Run `modal run triton_kernels/bench_chart.py` first, or pass `--sample` for synthetic data.")
            return

    create_chart(data, output_path)


if __name__ == "__main__":
    main()
