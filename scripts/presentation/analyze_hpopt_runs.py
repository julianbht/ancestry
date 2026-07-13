"""Analyze a wandb hpopt sweep dump: load every run's config + summary into a
table and probe whether the best trial's hyperparameters are a robust signal or
a lucky best-of-N outlier.

Reads run dirs of the form <wandb_dir>/run-*/files/{config.yaml,wandb-summary.json}.
Read-only; prints a report to stdout.

Usage:
    uv run python scripts/analyze_hpopt_runs.py [--wandb-dir podcopy/wandb]
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

# Objective used by the search (pipeline/experiments/face_recognition/search.py).
BETA = 0.5

PARAM_KEYS = [
    "model_pack",
    "det_size",
    "pad_ratio",
    "det_thresh",
    "distance_metric",
    "threshold",
    "threshold_normalized",
]
METRIC_KEYS = [
    "fbeta",
    "precision",
    "recall",
    "specificity",
    "n_emitted_labels",
    "n_known_queries",
    "n_negative_queries",
]


def load_runs(wandb_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(wandb_dir.glob("run-*")):
        cfg_path = run_dir / "files" / "config.yaml"
        sum_path = run_dir / "files" / "wandb-summary.json"
        if not cfg_path.exists() or not sum_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        summary = json.loads(sum_path.read_text())
        row = {"run": run_dir.name}
        for k in PARAM_KEYS:
            row[k] = cfg[k]["value"] if k in cfg else None
        for k in METRIC_KEYS:
            row[k] = summary.get(k)
        rows.append(row)
    return pd.DataFrame(rows)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wandb-dir", default="podcopy/wandb", type=Path)
    args = ap.parse_args()

    df = load_runs(args.wandb_dir)
    # Drop trials that never produced a metric (crashed / empty summary).
    df = df.dropna(subset=["fbeta"]).reset_index(drop=True)
    n = len(df)

    section(f"OVERVIEW — {n} trials with metrics")
    print(df[["fbeta", "precision", "recall", "specificity"]].describe().round(4).to_string())
    print("\ndistance_metric counts:\n", df["distance_metric"].value_counts().to_string())
    print("\nmodel_pack counts:\n", df["model_pack"].value_counts().to_string())

    best = df.sort_values("fbeta", ascending=False).reset_index(drop=True)
    best_fbeta = best.loc[0, "fbeta"]

    section("TOP 15 TRIALS BY FBETA")
    cols = ["fbeta", "precision", "recall", "specificity", "model_pack",
            "det_size", "pad_ratio", "det_thresh", "distance_metric",
            "threshold", "n_emitted_labels"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(best[cols].head(15).round(4).to_string())

    section("HOW PEAKED IS THE OPTIMUM? (selection-bias / overfitting gauge)")
    for delta in (0.005, 0.01, 0.02, 0.03):
        cnt = int((best["fbeta"] >= best_fbeta - delta).sum())
        print(f"  trials within {delta:.3f} of best ({best_fbeta:.4f}): {cnt}")
    print(f"  best={best_fbeta:.4f}  95th pct={best['fbeta'].quantile(0.95):.4f}  "
          f"90th pct={best['fbeta'].quantile(0.90):.4f}  median={best['fbeta'].median():.4f}")

    section("DET_THRESH MARGINAL EFFECT (all trials)")
    g = df.groupby("det_thresh")["fbeta"].agg(["count", "mean", "median", "max"]).round(4)
    print(g.to_string())

    section("DET_THRESH among the TOP 50 trials")
    top = best.head(50)
    print(top["det_thresh"].value_counts().sort_index().to_string())
    print(f"\n  top-50 det_thresh: mean={top['det_thresh'].mean():.3f} "
          f"median={top['det_thresh'].median():.3f}")

    # Hold the best trial's categorical family fixed and see if low det_thresh is
    # consistently better, or whether 0.1 just happened to win once.
    bp = best.loc[0]
    section(f"FIXED FAMILY: model_pack={bp.model_pack}, det_size={bp.det_size}, "
            f"distance_metric={bp.distance_metric}")
    fam = df[(df.model_pack == bp.model_pack)
             & (df.det_size == bp.det_size)
             & (df.distance_metric == bp.distance_metric)]
    fg = fam.groupby("det_thresh")["fbeta"].agg(["count", "mean", "median", "max"]).round(4)
    print(f"  {len(fam)} trials in this family")
    print(fg.to_string())

    section("OTHER PARAMS among TOP 50 (stability check)")
    for k in ["model_pack", "det_size", "distance_metric", "pad_ratio"]:
        print(f"\n{k}:\n", top[k].value_counts().sort_index().to_string())
    print(f"\nthreshold (top-50, {bp.distance_metric} family): "
          f"{top[top.distance_metric == bp.distance_metric]['threshold'].describe().round(3).to_dict()}")


if __name__ == "__main__":
    main()
