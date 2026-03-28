"""One-command pipeline for rPPG extraction, analytics, and statistics."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], title: str) -> None:
    print(f"\n[STEP] {title}")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Step failed ({title}), exit_code={proc.returncode}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full PREVERVECT rPPG pipeline.")
    p.add_argument("--real_dir", type=Path, default=Path("raw_data/original_sequences/youtube/c40/videos"))
    p.add_argument("--fake_dir", type=Path, default=Path("raw_data/manipulated_sequences/Deepfakes/c40/videos"))
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sample_every", type=int, default=2)
    p.add_argument("--real_csv", type=Path, default=Path("rppg_signals_real.csv"))
    p.add_argument("--fake_csv", type=Path, default=Path("rppg_signals_fake.csv"))
    p.add_argument("--real_metrics", type=Path, default=Path("video_metrics_real.csv"))
    p.add_argument("--fake_metrics", type=Path, default=Path("video_metrics_fake.csv"))
    p.add_argument("--report_dir", type=Path, default=Path("reports"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    run_step(
        [
            py,
            "advanced_extractor.py",
            "--real_dir",
            str(args.real_dir),
            "--fake_dir",
            str(args.fake_dir),
            "--real_out",
            str(args.real_csv),
            "--fake_out",
            str(args.fake_csv),
            "--workers",
            str(args.workers),
            "--sample_every",
            str(args.sample_every),
        ],
        "Extract rPPG signals",
    )

    run_step(
        [
            py,
            "signal_analytics.py",
            "--real_csv",
            str(args.real_csv),
            "--fake_csv",
            str(args.fake_csv),
            "--real_out",
            str(args.real_metrics),
            "--fake_out",
            str(args.fake_metrics),
        ],
        "Compute signal metrics",
    )

    run_step(
        [
            py,
            "stat_reporter.py",
            "--real_metrics",
            str(args.real_metrics),
            "--fake_metrics",
            str(args.fake_metrics),
            "--out_dir",
            str(args.report_dir),
            "--metric",
            "snr_db",
        ],
        "Generate statistical report",
    )

    print("\n[DONE] Full pipeline completed.")
    print(f"Outputs: {args.real_csv}, {args.fake_csv}, {args.real_metrics}, {args.fake_metrics}, {args.report_dir}")


if __name__ == "__main__":
    main()
