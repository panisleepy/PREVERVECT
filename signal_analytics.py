"""Signal processing and per-video metric extraction for rPPG."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, detrend, welch, filtfilt


def read_grouped_signals(csv_path: Path) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"time_sec": [], "pos_signal": []})
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["video_name"]
            t = float(row["time_sec"])
            s = float(row["pos_signal"])
            grouped[name]["time_sec"].append(t)
            grouped[name]["pos_signal"].append(s)

    out: dict[str, dict[str, np.ndarray]] = {}
    for name, d in grouped.items():
        out[name] = {
            "time_sec": np.asarray(d["time_sec"], dtype=np.float32),
            "pos_signal": np.asarray(d["pos_signal"], dtype=np.float32),
        }
    return out


def estimate_fs(time_sec: np.ndarray) -> float:
    if len(time_sec) < 2:
        return 30.0
    dt = np.diff(time_sec)
    med = float(np.median(dt[dt > 1e-6])) if np.any(dt > 1e-6) else 1 / 30.0
    fs = 1.0 / max(med, 1e-6)
    return float(np.clip(fs, 5.0, 120.0))


def preprocess_signal(sig: np.ndarray, fs: float) -> np.ndarray:
    if len(sig) < 8:
        return np.zeros_like(sig)
    x = detrend(sig.astype(np.float32), type="linear")
    nyq = fs * 0.5
    low = 0.7 / nyq
    high = min(3.0 / nyq, 0.99)
    if low >= high:
        return x
    b, a = butter(3, [low, high], btype="bandpass")
    y = filtfilt(b, a, x, method="gust")
    y = (y - np.mean(y)) / (np.std(y) + 1e-6)
    return y.astype(np.float32)


def peak_hr_bpm(sig: np.ndarray, fs: float) -> tuple[float, np.ndarray, np.ndarray]:
    if len(sig) < 16:
        return 0.0, np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    nperseg = max(16, min(256, len(sig)))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(band):
        return 0.0, freqs, psd
    f_band = freqs[band]
    p_band = psd[band]
    f_peak = float(f_band[np.argmax(p_band)])
    return f_peak * 60.0, freqs, psd


def snr_in_hr_band(freqs: np.ndarray, psd: np.ndarray, f0_hz: float, bw_hz: float = 0.1) -> float:
    if len(freqs) == 0:
        return -np.inf
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(band):
        return -np.inf
    signal_band = band & (freqs >= max(0.7, f0_hz - bw_hz)) & (freqs <= min(3.0, f0_hz + bw_hz))
    noise_band = band & (~signal_band)
    p_sig = float(np.sum(psd[signal_band]))
    p_noise = float(np.sum(psd[noise_band])) + 1e-9
    return 10.0 * np.log10((p_sig + 1e-9) / p_noise)


def spectral_centroid(freqs: np.ndarray, psd: np.ndarray) -> float:
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(band):
        return 0.0
    f = freqs[band]
    p = psd[band]
    return float(np.sum(f * p) / (np.sum(p) + 1e-9))


def bpm_stability(sig: np.ndarray, fs: float, win_sec: float = 10.0, hop_sec: float = 5.0) -> float:
    win = int(max(8, round(win_sec * fs)))
    hop = int(max(4, round(hop_sec * fs)))
    if len(sig) < win:
        return 0.0
    bpms = []
    for st in range(0, len(sig) - win + 1, hop):
        seg = sig[st : st + win]
        bpm, _, _ = peak_hr_bpm(seg, fs)
        if bpm > 0:
            bpms.append(bpm)
    if len(bpms) < 2:
        return 0.0
    return float(np.std(np.asarray(bpms)))


def analyze_csv(input_csv: Path, output_csv: Path, label: str) -> None:
    grouped = read_grouped_signals(input_csv)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video_name",
                "label",
                "n_samples",
                "fs",
                "peak_bpm",
                "snr_db",
                "spectral_centroid_hz",
                "bpm_std",
            ],
        )
        writer.writeheader()

        for name, d in grouped.items():
            t = d["time_sec"]
            s = d["pos_signal"]
            fs = estimate_fs(t)
            x = preprocess_signal(s, fs)
            bpm, freqs, psd = peak_hr_bpm(x, fs)
            f0 = bpm / 60.0 if bpm > 0 else 1.2
            snr = snr_in_hr_band(freqs, psd, f0_hz=f0)
            cent = spectral_centroid(freqs, psd)
            bpm_std = bpm_stability(x, fs)
            writer.writerow(
                {
                    "video_name": name,
                    "label": label,
                    "n_samples": len(s),
                    "fs": fs,
                    "peak_bpm": bpm,
                    "snr_db": snr,
                    "spectral_centroid_hz": cent,
                    "bpm_std": bpm_std,
                }
            )

    print(f"[DONE] Metrics saved to {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="rPPG signal analytics.")
    p.add_argument("--real_csv", type=Path, default=Path("rppg_signals_real.csv"))
    p.add_argument("--fake_csv", type=Path, default=Path("rppg_signals_fake.csv"))
    p.add_argument("--real_out", type=Path, default=Path("video_metrics_real.csv"))
    p.add_argument("--fake_out", type=Path, default=Path("video_metrics_fake.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyze_csv(args.real_csv, args.real_out, label="Real")
    analyze_csv(args.fake_csv, args.fake_out, label="Fake")


if __name__ == "__main__":
    main()
