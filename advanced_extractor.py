"""Advanced rPPG extraction using MediaPipe Face Mesh and POS."""

from __future__ import annotations

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg"}

# Landmark sets are intentionally compact for robust polygon masks.
FOREHEAD_IDX = [10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 119, 120, 100, 126, 142, 203, 205, 50, 36]
RIGHT_CHEEK_IDX = [346, 347, 348, 349, 329, 355, 371, 423, 425, 280, 266]


def collect_videos(data_root: Path) -> list[Path]:
    videos: list[Path] = []
    for p in data_root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    return sorted(videos)


def _roi_weighted_mean(frame_bgr: np.ndarray, polygon_xy: np.ndarray) -> tuple[float, float, float, float]:
    h, w = frame_bgr.shape[:2]
    if polygon_xy.shape[0] < 3:
        return 0.0, 0.0, 0.0, 0.0

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, polygon_xy.astype(np.int32), 255)
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return 0.0, 0.0, 0.0, 0.0

    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    radius2 = float(np.var(xs) + np.var(ys) + 1e-6)
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * radius2))
    weights_sum = float(np.sum(weights)) + 1e-6

    pix = frame_bgr[ys, xs].astype(np.float32)
    # Convert BGR -> RGB order.
    b = float(np.sum(pix[:, 0] * weights) / weights_sum)
    g = float(np.sum(pix[:, 1] * weights) / weights_sum)
    r = float(np.sum(pix[:, 2] * weights) / weights_sum)
    return r, g, b, 1.0


def _extract_polygons(landmarks, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def pts(indices: list[int]) -> np.ndarray:
        out = []
        for i in indices:
            lm = landmarks[i]
            out.append((int(lm.x * width), int(lm.y * height)))
        return np.asarray(out, dtype=np.int32)

    return pts(FOREHEAD_IDX), pts(LEFT_CHEEK_IDX), pts(RIGHT_CHEEK_IDX)


def _fill_missing(rgb_trace: np.ndarray) -> np.ndarray:
    out = rgb_trace.copy()
    for c in range(3):
        x = out[:, c]
        idx = np.arange(len(x))
        valid = np.isfinite(x)
        if np.sum(valid) < 2:
            out[:, c] = 0.0
        else:
            out[:, c] = np.interp(idx, idx[valid], x[valid])
    return out


def pos_projection(rgb_trace: np.ndarray, fps: float, win_sec: float = 1.6) -> np.ndarray:
    """
    POS algorithm with overlap-add reconstruction.

    Args:
        rgb_trace: [T, 3] in RGB order.
        fps: frame rate.
    """
    T = rgb_trace.shape[0]
    if T == 0:
        return np.zeros(0, dtype=np.float32)
    win = max(8, int(round(win_sec * fps)))
    if T < win:
        win = max(4, T)

    X = rgb_trace.astype(np.float32)
    s = np.zeros(T, dtype=np.float32)
    W = np.zeros(T, dtype=np.float32)

    P = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float32)
    for n in range(0, T - win + 1):
        seg = X[n : n + win]
        mu = np.mean(seg, axis=0) + 1e-6
        Cn = (seg / mu) - 1.0
        S = P @ Cn.T  # [2, win]
        s1 = S[0]
        s2 = S[1]
        alpha = float(np.std(s1) / (np.std(s2) + 1e-6))
        h = s1 - alpha * s2
        h = h - np.mean(h)
        s[n : n + win] += h
        W[n : n + win] += 1.0

    valid = W > 0
    s[valid] = s[valid] / W[valid]
    if np.std(s) > 1e-6:
        s = (s - np.mean(s)) / (np.std(s) + 1e-6)
    return s


def process_video(video_path: Path, sample_every: int = 2) -> list[dict[str, float | str | int]]:
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 1e-3:
        fps = 30.0

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows: list[dict[str, float | str | int]] = []
    rgb_trace = []
    frame_ids = []
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every != 0:
            frame_idx += 1
            continue

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        r_mean, g_mean, b_mean = np.nan, np.nan, np.nan
        conf = 0.0
        if res.multi_face_landmarks:
            lm = res.multi_face_landmarks[0].landmark
            p_fore, p_left, p_right = _extract_polygons(lm, w, h)
            r1, g1, b1, c1 = _roi_weighted_mean(frame, p_fore)
            r2, g2, b2, c2 = _roi_weighted_mean(frame, p_left)
            r3, g3, b3, c3 = _roi_weighted_mean(frame, p_right)
            valid = c1 + c2 + c3
            if valid > 0:
                r_mean = (r1 + r2 + r3) / valid
                g_mean = (g1 + g2 + g3) / valid
                b_mean = (b1 + b2 + b3) / valid
                conf = min(1.0, valid / 3.0)

        rows.append(
            {
                "video_name": video_path.name,
                "frame_idx": frame_idx,
                "time_sec": frame_idx / fps,
                "R": float(r_mean) if np.isfinite(r_mean) else np.nan,
                "G": float(g_mean) if np.isfinite(g_mean) else np.nan,
                "B": float(b_mean) if np.isfinite(b_mean) else np.nan,
                "face_confidence": float(conf),
            }
        )
        rgb_trace.append([r_mean, g_mean, b_mean])
        frame_ids.append(frame_idx)
        frame_idx += 1

    cap.release()
    face_mesh.close()

    if not rows:
        return rows

    rgb_arr = np.asarray(rgb_trace, dtype=np.float32)
    rgb_arr = _fill_missing(rgb_arr)
    pos_sig = pos_projection(rgb_arr, fps=fps)
    for i in range(len(rows)):
        rows[i]["pos_signal"] = float(pos_sig[i]) if i < len(pos_sig) else 0.0
    return rows


def write_rows(csv_path: Path, rows: list[dict[str, float | str | int]], write_header: bool = False) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["video_name", "frame_idx", "time_sec", "R", "G", "B", "pos_signal", "face_confidence"],
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def run_split(input_dir: Path, output_csv: Path, workers: int, sample_every: int) -> None:
    videos = collect_videos(input_dir)
    if not videos:
        print(f"[INFO] No videos found in: {input_dir}")
        return

    if output_csv.exists():
        output_csv.unlink()

    write_rows(output_csv, [], write_header=True)

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_video, v, sample_every): v for v in videos}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Extracting {input_dir.name}"):
            rows = fut.result()
            if rows:
                write_rows(output_csv, rows, write_header=False)

    print(f"[DONE] Saved {output_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Advanced rPPG extractor (Face Mesh + POS).")
    p.add_argument("--real_dir", type=Path, default=Path("raw_data/original_sequences/youtube/c40/videos"))
    p.add_argument("--fake_dir", type=Path, default=Path("raw_data/manipulated_sequences/Deepfakes/c40/videos"))
    p.add_argument("--real_out", type=Path, default=Path("rppg_signals_real.csv"))
    p.add_argument("--fake_out", type=Path, default=Path("rppg_signals_fake.csv"))
    p.add_argument("--workers", type=int, default=max(1, (os_cpu_count() // 2)))
    p.add_argument("--sample_every", type=int, default=2)
    return p.parse_args()


def os_cpu_count() -> int:
    try:
        import os

        return int(os.cpu_count() or 4)
    except Exception:
        return 4


def main() -> None:
    args = parse_args()
    run_split(args.real_dir, args.real_out, workers=args.workers, sample_every=args.sample_every)
    run_split(args.fake_dir, args.fake_out, workers=args.workers, sample_every=args.sample_every)


if __name__ == "__main__":
    main()
