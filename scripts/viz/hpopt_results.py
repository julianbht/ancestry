"""Visualise W&B hpopt CSV exports for frame_crop SAM experiments.

Usage:
    uv run python scripts/viz/hpopt_results.py <csv_path>
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt1
import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: uv run python scripts/viz/hpopt_results.py <csv_path>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    df = pd.read_csv(csv_path)

    summary = df.groupby("prompt")["detection_rate"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary.index, summary.values)
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean detection_rate")
    ax.set_title("SAM — mean detection_rate by prompt")
    fig.tight_layout()

    out_path = csv_path.parent / "sam_prompt_bar.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
