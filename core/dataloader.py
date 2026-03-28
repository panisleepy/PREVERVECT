"""Video-first dataloader for zero-storage training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.fft_tools import power_spectrum_shifted_bgr


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".mpeg"}


@dataclass
class VideoSample:
    path: Path
    label: int  # Real=0, Fake=1
    frames: int


def infer_label_from_path(video_path: Path) -> int:
    text = str(video_path).lower()
    if "real" in text or "original" in text:
        return 0
    return 1


def collect_video_samples(data_root: Path) -> list[VideoSample]:
    samples: list[VideoSample] = []
    for p in data_root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXTS:
            continue
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            continue
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            samples.append(VideoSample(path=p, label=infer_label_from_path(p), frames=frames))
    return samples


def split_video_samples(
    samples: list[VideoSample],
    val_ratio: float,
    seed: int,
) -> tuple[list[VideoSample], list[VideoSample]]:
    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    val_samples = shuffled[:val_size]
    train_samples = shuffled[val_size:]
    return train_samples, val_samples


def _jpeg_compress(frame_bgr: np.ndarray, quality: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return frame_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else frame_bgr


def _face_occlusion(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    occ_w = int(w * rng.uniform(0.12, 0.28))
    occ_h = int(h * rng.uniform(0.10, 0.25))
    x0 = int(rng.uniform(0, max(1, w - occ_w)))
    y0 = int(rng.uniform(0, max(1, h - occ_h)))
    color = tuple(int(c) for c in rng.choices(range(20, 210), k=3))
    out = frame_bgr.copy()
    cv2.rectangle(out, (x0, y0), (x0 + occ_w, y0 + occ_h), color, -1)
    return out


def _random_downscale_upscale(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate bitrate / resolution loss then re-encode look (breaks 'sharp=real' shortcuts)."""
    h, w = frame_bgr.shape[:2]
    scale = rng.uniform(0.42, 0.92)
    nh, nw = max(8, int(h * scale)), max(8, int(w * scale))
    small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _maybe_gaussian_blur(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    k = rng.choice([3, 5, 7])
    sigma = rng.uniform(0.4, 2.2)
    return cv2.GaussianBlur(frame_bgr, (k, k), sigmaX=sigma, sigmaY=sigma)


def _rotate_bgr_reflect(frame_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    center = (w * 0.5, h * 0.5)
    m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    return cv2.warpAffine(
        frame_bgr,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _color_jitter_bgr(frame_bgr: np.ndarray, rng: random.Random, strength: float = 0.2) -> np.ndarray:
    """Brightness / contrast / saturation jitter (~ torchvision ColorJitter) on BGR uint8."""
    img = frame_bgr.astype(np.float32)
    c = rng.uniform(1.0 - strength, 1.0 + strength)
    b = rng.uniform(-strength, strength) * 255.0
    img = img * c + b
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    s_scale = rng.uniform(1.0 - strength, 1.0 + strength)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _train_geom_color_augment_bgr(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Spatially consistent augmentations for the RGB stream.

    FFT (`power_spectrum_shifted_bgr`) must be computed *after* this so the
    spectrum matches the warped / color-jittered frame.
    """
    out = frame_bgr
    if rng.random() < 0.5:
        out = cv2.flip(out, 1)
    angle = rng.uniform(-10.0, 10.0)
    out = _rotate_bgr_reflect(out, angle)
    out = _color_jitter_bgr(out, rng, strength=0.2)
    if rng.random() < 0.2:
        k = rng.choice([3, 5])
        sigma = rng.uniform(0.5, 1.5)
        out = cv2.GaussianBlur(out, (k, k), sigmaX=sigma, sigmaY=sigma)
    return out


def augment_frame(frame_bgr: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Train-time augmentations.

    Heavy emphasis on *quality / resolution / blur* jitter so the model cannot
    equate compression or sharpness with class label (a common failure mode
    when FaceForensics splits mix qualities or when test domain differs).
    """
    out = frame_bgr.copy()
    # Resolution / pipeline artifacts (often applied — domain randomization)
    if rng.random() < 0.55:
        out = _random_downscale_upscale(out, rng)
    if rng.random() < 0.45:
        out = _maybe_gaussian_blur(out, rng)
    if rng.random() < 0.48:
        sigma = rng.uniform(3.0, 12.0)
        noise = rng.normalvariate(0.0, sigma)
        gauss = np.random.normal(loc=noise, scale=sigma, size=out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + gauss, 0, 255).astype(np.uint8)
    # JPEG: wide quality range, independent of label
    if rng.random() < 0.55:
        out = _jpeg_compress(out, quality=rng.randint(12, 88))
    if rng.random() < 0.28:
        out = _face_occlusion(out, rng)
    return out


class VideoFramePairDataset(Dataset):
    """
    Dynamic frame loader from videos.

    It samples frames on-the-fly from videos and never stores extracted images.
    """

    def __init__(
        self,
        videos: list[VideoSample],
        image_size: int = 224,
        frames_per_video: int = 16,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        self.videos = videos
        self.augment = augment
        self.seed = seed
        self.index_map: list[tuple[int, int]] = []
        for vid_idx, v in enumerate(videos):
            n = max(1, min(frames_per_video, v.frames))
            step = max(v.frames // n, 1)
            for j in range(n):
                frame_idx = min(j * step, v.frames - 1)
                self.index_map.append((vid_idx, frame_idx))

        self.rgb_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.fft_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((image_size, image_size), antialias=True),
                transforms.Normalize(
                    mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5),
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def _read_video_frame(self, path: Path, frame_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise ValueError(f"Cannot decode frame {frame_idx} from {path}")
        return frame

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vid_idx, frame_idx = self.index_map[idx]
        v = self.videos[vid_idx]

        frame = self._read_video_frame(v.path, frame_idx)
        rng = random.Random(self.seed + idx * 31 + frame_idx)
        if self.augment:
            # Domain / compression jitter first, then geom+color (FFT follows so streams stay aligned).
            frame = augment_frame(frame, rng)
            frame = _train_geom_color_augment_bgr(frame, rng)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        spectrum = power_spectrum_shifted_bgr(frame)

        rgb_tensor = self.rgb_transform(rgb)
        fft_tensor = self.fft_transform(spectrum)
        target = torch.tensor([v.label], dtype=torch.float32)
        return rgb_tensor, fft_tensor, target


# Alias for notebooks / external scripts expecting this name.
DeepfakeDataset = VideoFramePairDataset
