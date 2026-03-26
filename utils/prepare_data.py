"""Prepare FaceForensics++ videos into labeled frame folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg"}


def infer_label_from_path(video_path: Path) -> str:
    text = str(video_path).lower()
    if "real" in text or "original" in text:
        return "Real"
    return "Fake"


def extract_frames(
    video_path: Path,
    output_root: Path,
    every_n: int = 10,
    max_frames: int = 0,
) -> int:
    label = infer_label_from_path(video_path)
    dst_dir = output_root / label / video_path.stem
    dst_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    saved = 0
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            frame_name = dst_dir / f"{saved:06d}.jpg"
            cv2.imwrite(str(frame_name), frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        idx += 1

    cap.release()
    return saved


def collect_videos(data_root: Path) -> list[Path]:
    return [p for p in data_root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract FaceForensics++ frames into Real/Fake folders.")
    parser.add_argument("--data_root", type=Path, required=True, help="Path to FaceForensics++ root.")
    parser.add_argument("--output_root", type=Path, default=Path("data/frames"), help="Output root for extracted frames.")
    parser.add_argument("--every_n", type=int, default=10, help="Save one frame every N frames.")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames per video, 0 means unlimited.")
    args = parser.parse_args()

    videos = collect_videos(args.data_root)
    if not videos:
        print(f"[INFO] No videos found in: {args.data_root}")
        return

    total = 0
    for i, video in enumerate(videos, start=1):
        n = extract_frames(video, args.output_root, every_n=args.every_n, max_frames=args.max_frames)
        total += n
        print(f"[{i}/{len(videos)}] {video.name} -> {n} frames")

    print(f"[DONE] Extracted {total} frames from {len(videos)} videos into: {args.output_root}")


if __name__ == "__main__":
    main()
