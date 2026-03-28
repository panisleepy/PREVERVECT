"""Statistical reporting for Real vs Fake rPPG metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu


def read_metric_column(csv_path: Path, column: str) -> np.ndarray:
    vals = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row[column])
                if np.isfinite(v):
                    vals.append(v)
            except (KeyError, ValueError):
                continue
    return np.asarray(vals, dtype=np.float32)


def summarize(x: np.ndarray) -> dict[str, float]:
    if len(x) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0}
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
    }


def violin_plot(real: np.ndarray, fake: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    parts = plt.violinplot([real, fake], showmeans=True, showmedians=True)
    for i, b in enumerate(parts["bodies"]):
        b.set_alpha(0.65)
        b.set_facecolor("#6aaed6" if i == 0 else "#e48ca8")
    plt.xticks([1, 2], ["Real", "Fake"])
    plt.ylabel("SNR (dB)")
    plt.title("SNR Distribution Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Statistical report generator for rPPG metrics.")
    p.add_argument("--real_metrics", type=Path, default=Path("video_metrics_real.csv"))
    p.add_argument("--fake_metrics", type=Path, default=Path("video_metrics_fake.csv"))
    p.add_argument("--out_dir", type=Path, default=Path("reports"))
    p.add_argument("--metric", type=str, default="snr_db")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    real = read_metric_column(args.real_metrics, args.metric)
    fake = read_metric_column(args.fake_metrics, args.metric)
    if len(real) == 0 or len(fake) == 0:
        raise RuntimeError("Empty metric arrays. Run signal_analytics.py first.")

    stat = mannwhitneyu(real, fake, alternative="two-sided")
    violin_plot(real, fake, args.out_dir / f"violin_{args.metric}.png")

    summary = {
        "metric": args.metric,
        "real": summarize(real),
        "fake": summarize(fake),
        "mann_whitney_u": float(stat.statistic),
        "p_value": float(stat.pvalue),
    }

    with (args.out_dir / "stat_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (args.out_dir / "stat_summary.txt").open("w", encoding="utf-8") as f:
        f.write("rPPG Statistical Report\n")
        f.write("=======================\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"Real  -> n={summary['real']['n']}, mean={summary['real']['mean']:.4f}, std={summary['real']['std']:.4f}\n")
        f.write(f"Fake  -> n={summary['fake']['n']}, mean={summary['fake']['mean']:.4f}, std={summary['fake']['std']:.4f}\n")
        f.write(f"Mann-Whitney U: {summary['mann_whitney_u']:.4f}\n")
        f.write(f"p-value: {summary['p_value']:.6e}\n")
        if summary["p_value"] < 0.05:
            f.write("Result: significant difference (p < 0.05)\n")
        else:
            f.write("Result: not significant (p >= 0.05)\n")

    print(f"[DONE] Report generated in {args.out_dir}")


if __name__ == "__main__":
    main()
