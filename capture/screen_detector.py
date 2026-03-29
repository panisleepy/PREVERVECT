"""Realtime PREVERVECT detector with threaded inference and glass UI."""

from __future__ import annotations

import argparse
import csv
import math
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import sys
import urllib.request

import cv2
import mediapipe as mp
import mss
import numpy as np
import torch
from scipy.signal import butter, filtfilt, welch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model import build_specxnet
from utils.fft_tools import power_spectrum_shifted_bgr


LIVE_WINDOW_NAME = "PREVERVECT Live Feed"
CONSOLE_WINDOW_NAME = "PREVERVECT Console"
PANEL_WIDTH = 430
PANEL_HEIGHT = 560
MODEL_SIZE = 224

# Scoring: linear calibration reduces oversensitivity to fakes (clip 0–1).
SCORE_CALIB_OFFSET = 0.3
SCORE_CALIB_DIV = 0.7
SCORE_SMOOTH_LEN = 15

# Laplacian variance (focus); below → low quality → force 50/50 DFA fusion.
LAPLACIAN_VAR_THRESHOLD = 120.0

# BPM: SNR = energy near 1.2 Hz / energy in rest of 0.5–4 Hz band; below → unreliable.
BPM_SNR_12HZ_THRESHOLD = 1.5

# Palette: #FFD1DC, #AEC6CF, #F5F5DC in BGR.
SOFT_PINK = (220, 209, 255)
PASTEL_BLUE = (207, 198, 174)
BEIGE = (220, 245, 245)
SOFT_WHITE = (245, 245, 245)

FOREHEAD_IDX = [10, 67, 103, 109, 338, 297, 332, 284]
LEFT_CHEEK_IDX = [117, 118, 119, 120, 100, 126, 142, 203, 205, 50, 36]
RIGHT_CHEEK_IDX = [346, 347, 348, 349, 329, 355, 371, 423, 425, 280, 266]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime PREVERVECT screen detector")
    parser.add_argument("--target_window", type=str, default="", help="Window title keyword to capture.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Session CSV output folder.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="SpecXNet state_dict (.pth). If omitted, tries weights/specxnet_best.pth under project root.",
    )
    return parser.parse_args()


def _get_monitor_from_window_title(title_keyword: str) -> dict[str, int] | None:
    if not title_keyword:
        return None
    try:
        import pygetwindow as gw
    except Exception:
        return None
    keyword = title_keyword.lower()
    candidates = [w for w in gw.getAllWindows() if w.title and keyword in w.title.lower()]
    if not candidates:
        return None
    best = max(candidates, key=lambda w: max(0, w.width) * max(0, w.height))
    if best.width <= 0 or best.height <= 0:
        return None
    return {"top": int(best.top), "left": int(best.left), "width": int(best.width), "height": int(best.height)}


def _mcdm_preprocess_placeholder(face_roi: np.ndarray) -> np.ndarray:
    return face_roi


def _calibrate_fake_score(raw: float) -> float:
    x = (float(raw) - SCORE_CALIB_OFFSET) / SCORE_CALIB_DIV
    return float(np.clip(x, 0.0, 1.0))


def _unsharp_mask_bgr(bgr: np.ndarray, amount: float = 0.85, sigma: float = 1.2) -> np.ndarray:
    if bgr.size == 0:
        return bgr
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharp = cv2.addWeighted(bgr, 1.0 + amount, blurred, -amount, 0.0)
    return np.clip(sharp, 0, 255).astype(np.uint8)


def _laplacian_var_bgr(bgr: np.ndarray) -> float:
    if bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _expand_bbox_scale(
    x0: int, y0: int, x1: int, y1: int, w: int, h: int, scale: float = 1.2
) -> tuple[int, int, int, int]:
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    nw = bw * scale
    nh = bh * scale
    x0n = int(round(cx - nw * 0.5))
    y0n = int(round(cy - nh * 0.5))
    x1n = int(round(cx + nw * 0.5))
    y1n = int(round(cy + nh * 0.5))
    return _clamp_bbox(x0n, y0n, x1n, y1n, w, h)


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    return x0, y0, x1, y1


def _poly_from_landmarks(landmarks, indices: list[int], w: int, h: int) -> np.ndarray:
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
    return np.asarray(pts, dtype=np.int32)


def _weighted_mean_rgb(frame_bgr: np.ndarray, poly: np.ndarray) -> tuple[float, float, float]:
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, poly, 255)
    ys, xs = np.where(mask > 0)
    if len(xs) < 8:
        return np.nan, np.nan, np.nan
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    dx = xs.astype(np.float32) - cx
    dy = ys.astype(np.float32) - cy
    sigma2 = float(np.var(xs) + np.var(ys) + 1e-6)
    weights = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma2))
    p = frame_bgr[ys, xs].astype(np.float32)
    denom = float(np.sum(weights)) + 1e-6
    b = float(np.sum(p[:, 0] * weights) / denom)
    g = float(np.sum(p[:, 1] * weights) / denom)
    r = float(np.sum(p[:, 2] * weights) / denom)
    return r, g, b


def _to_model_tensor_rgb(face_bgr: np.ndarray) -> torch.Tensor:
    # Match core/dataloader VideoFramePairDataset: ToTensor → Resize 224 → ImageNet normalize.
    h, w = face_bgr.shape[:2]
    interp = cv2.INTER_AREA if h >= MODEL_SIZE and w >= MODEL_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(face_bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std
    return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)


def _to_model_tensor_fft(face_bgr: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
    spec = power_spectrum_shifted_bgr(face_bgr)
    sh, sw = spec.shape[:2]
    interp = cv2.INTER_AREA if sh >= MODEL_SIZE and sw >= MODEL_SIZE else cv2.INTER_LINEAR
    spec_resized = cv2.resize(spec, (MODEL_SIZE, MODEL_SIZE), interpolation=interp).astype(np.float32)
    norm = (spec_resized - 0.5) / 0.5
    tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)
    thumb = (spec_resized * 255.0).clip(0, 255).astype(np.uint8)
    return tensor, thumb


def _pos_signal(rgb_trace: np.ndarray, fps: float, win_sec: float = 1.6) -> np.ndarray:
    t = len(rgb_trace)
    if t < 4:
        return np.zeros(t, dtype=np.float32)
    win = max(4, int(round(win_sec * fps)))
    win = min(win, t)
    p = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float32)
    s = np.zeros(t, dtype=np.float32)
    wsum = np.zeros(t, dtype=np.float32)
    for n in range(0, t - win + 1):
        seg = rgb_trace[n : n + win]
        mu = np.mean(seg, axis=0) + 1e-6
        cn = (seg / mu) - 1.0
        proj = p @ cn.T
        alpha = float(np.std(proj[0]) / (np.std(proj[1]) + 1e-6))
        h = proj[0] - alpha * proj[1]
        h = h - np.mean(h)
        s[n : n + win] += h
        wsum[n : n + win] += 1.0
    valid = wsum > 0
    s[valid] = s[valid] / wsum[valid]
    if np.std(s) > 1e-6:
        s = (s - np.mean(s)) / (np.std(s) + 1e-6)
    return s


def _filter_pos(sig: np.ndarray, fps: float) -> np.ndarray:
    if len(sig) < 24:
        return sig
    nyq = 0.5 * fps
    low = 0.7 / max(nyq, 1e-6)
    high = min(3.0 / max(nyq, 1e-6), 0.99)
    if low >= high:
        return sig
    b, a = butter(2, [low, high], btype="bandpass")
    padlen = 3 * (max(len(a), len(b)) - 1)
    if len(sig) <= padlen:
        return sig
    return filtfilt(b, a, sig).astype(np.float32)


def _estimate_bpm_and_snr_metrics(
    sig: np.ndarray, fs: float
) -> tuple[float, float, float, bool]:
    """
    BPM from Welch peak in 0.7–3 Hz; welch_snr_db = 10*log10(peak band / rest in HR band);
    snr_12hz_ratio = P(1.1–1.3 Hz) / (P(0.5–4 Hz) - P(1.1–1.3 Hz)); reliable if ratio >= threshold.
    """
    if len(sig) < 16 or fs < 1e-3:
        return 0.0, -np.inf, 0.0, False
    nperseg = min(256, len(sig))
    freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
    band = (freqs >= 0.7) & (freqs <= 3.0)
    if not np.any(band):
        return 0.0, -np.inf, 0.0, False
    f_band = freqs[band]
    p_band = psd[band]
    peak_idx = int(np.argmax(p_band))
    f0 = float(f_band[peak_idx])
    bpm = f0 * 60.0
    sig_band = band & (freqs >= f0 - 0.1) & (freqs <= f0 + 0.1)
    noise_band = band & (~sig_band)
    p_sig = float(np.sum(psd[sig_band])) + 1e-9
    p_noise = float(np.sum(psd[noise_band])) + 1e-9
    welch_snr_db = 10.0 * math.log10(p_sig / p_noise)

    band_full = (freqs >= 0.5) & (freqs <= 4.0)
    p_full = float(np.sum(psd[band_full])) + 1e-12
    band_12 = (freqs >= 1.1) & (freqs <= 1.3)
    p_12 = float(np.sum(psd[band_12]))
    rest = max(p_full - p_12, 1e-12)
    snr_12hz_ratio = p_12 / rest
    is_reliable = snr_12hz_ratio >= BPM_SNR_12HZ_THRESHOLD
    return bpm, welch_snr_db, snr_12hz_ratio, is_reliable


class InferenceWorker(threading.Thread):
    """Threaded worker for Face Mesh tracking, SpecXNet inference, and POS rPPG."""

    def __init__(self, device: torch.device, weights_path: Path | None = None) -> None:
        super().__init__(daemon=True)
        self._device = device
        self._weights_path = weights_path
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._running = True
        self._result: dict = {
            "face_found": False,
            "bbox": None,
            "raw_fake_score": 0.5,
            "calib_fake_score": _calibrate_fake_score(0.5),
            "avg_fake_score": 0.5,
            "fake_prob": 0.5,
            "fake_prob_smooth": 0.5,
            "latency_ms": 0.0,
            "waveform": np.zeros(120, dtype=np.float32),
            "bpm": 0.0,
            "snr": -np.inf,
            "snr_12hz_ratio": 0.0,
            "is_bpm_reliable": False,
            "fft_thumb": np.zeros((64, 64, 3), dtype=np.uint8),
            "status": "Scanning...",
            "stability": 0.0,
        }
        self._model = build_specxnet(pretrained=True, device=device)
        self._load_weights()
        self._model.eval()
        self._tasks_detector = None
        self._tracker_mode = "center"
        self._init_face_tracker()
        self._calib_hist: deque[float] = deque(maxlen=SCORE_SMOOTH_LEN)
        self._rgb_trace: deque[np.ndarray] = deque(maxlen=240)
        self._fps_trace: deque[float] = deque(maxlen=50)
        self._frame_dt_hist: deque[float] = deque(maxlen=90)
        self._last_frame_arrival_t: float | None = None

    def _load_weights(self) -> None:
        candidates: list[Path] = []
        if self._weights_path is not None:
            wp = Path(self._weights_path)
            if wp.is_absolute():
                candidates.append(wp)
            else:
                candidates.append(PROJECT_ROOT / wp)
                candidates.append(Path.cwd() / wp)
        candidates.extend(
            [
                PROJECT_ROOT / "weights/specxnet_best.pth",
                PROJECT_ROOT / "weights/specxnet_latest.pth",
                PROJECT_ROOT / "weights/specxnet_finetuned.pt",
                Path("weights/specxnet_best.pth"),
                Path("weights/specxnet_latest.pth"),
                Path("weights/specxnet_finetuned.pt"),
            ]
        )
        seen: set[str] = set()
        for p in candidates:
            try:
                p = p.resolve()
            except OSError:
                continue
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            if p.is_file():
                try:
                    state = torch.load(p, map_location=self._device, weights_only=True)
                except TypeError:
                    state = torch.load(p, map_location=self._device)
                self._model.load_state_dict(state, strict=False)
                print(f"[INFO] Loaded model weights: {p}")
                return
        print("[WARN] No fine-tuned SpecXNet weights found. Using backbone defaults.")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        with self._lock:
            self._latest_frame = frame_bgr.copy()

    def get_result(self) -> dict:
        with self._lock:
            out = dict(self._result)
            out["waveform"] = self._result["waveform"].copy()
            out["fft_thumb"] = self._result["fft_thumb"].copy()
            return out

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        while self._running:
            with self._lock:
                frame = None if self._latest_frame is None else self._latest_frame.copy()
                self._latest_frame = None
            if frame is None:
                time.sleep(0.002)
                continue
            t_arrival = time.perf_counter()
            if self._last_frame_arrival_t is not None:
                dtf = t_arrival - self._last_frame_arrival_t
                if dtf > 1e-4:
                    self._frame_dt_hist.append(dtf)
            self._last_frame_arrival_t = t_arrival
            fs_sample = 30.0
            if len(self._frame_dt_hist) >= 5:
                mean_dt = float(np.mean(self._frame_dt_hist))
                fs_sample = 1.0 / max(mean_dt, 1e-3)
                fs_sample = float(np.clip(fs_sample, 8.0, 120.0))

            t0 = time.perf_counter()
            res = self._process_frame(frame, fs_sample=fs_sample)
            dt = max(time.perf_counter() - t0, 1e-6)
            fps = 1.0 / dt
            self._fps_trace.append(fps)
            res["stability"] = float(np.std(np.asarray(self._fps_trace))) if len(self._fps_trace) > 3 else 0.0
            res["latency_ms"] = dt * 1000.0
            with self._lock:
                self._result.update(res)
        if self._tasks_detector is not None:
            self._tasks_detector.close()

    def _ensure_task_model(self) -> Path | None:
        model_path = Path("weights/blaze_face_short_range.tflite")
        if model_path.exists():
            return model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/face_detector/"
            "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
        )
        try:
            print(f"[INFO] Downloading MediaPipe task model: {url}")
            urllib.request.urlretrieve(url, str(model_path))
            return model_path
        except Exception as e:
            print(f"[WARN] Failed to download task model, fallback to center ROI: {e}")
            return None

    def _init_face_tracker(self) -> None:
        try:
            if not hasattr(mp, "tasks"):
                print("[WARN] mediapipe.tasks unavailable, fallback to center ROI tracker.")
                return
            model_path = self._ensure_task_model()
            if model_path is None:
                return
            BaseOptions = mp.tasks.BaseOptions
            vision = mp.tasks.vision
            options = vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=0.5,
                running_mode=vision.RunningMode.IMAGE,
            )
            self._tasks_detector = vision.FaceDetector.create_from_options(options)
            self._tracker_mode = "tasks_detector"
            print("[INFO] Using mediapipe.tasks face detector.")
        except Exception as e:
            print(f"[WARN] Face tracker init failed, fallback to center ROI tracker: {e}")
            self._tasks_detector = None
            self._tracker_mode = "center"

    def _avg_calib(self) -> float:
        if not self._calib_hist:
            return 0.5
        return float(np.mean(self._calib_hist))

    def _detect_bbox_tasks(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        if self._tasks_detector is None:
            return None
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        det_result = self._tasks_detector.detect(mp_image)
        if not det_result.detections:
            return None
        best = max(det_result.detections, key=lambda d: d.categories[0].score if d.categories else 0.0)
        bb = best.bounding_box
        x0 = int(bb.origin_x)
        y0 = int(bb.origin_y)
        x1 = int(bb.origin_x + bb.width)
        y1 = int(bb.origin_y + bb.height)
        return _expand_bbox_scale(x0, y0, x1, y1, w, h, scale=1.2)

    def _process_frame(self, frame_bgr: np.ndarray, fs_sample: float) -> dict:
        h, w = frame_bgr.shape[:2]
        avg_prev = self._avg_calib()

        def _idle(pulse: float) -> dict:
            return {
                "face_found": False,
                "bbox": None,
                "raw_fake_score": 0.5,
                "calib_fake_score": _calibrate_fake_score(0.5),
                "avg_fake_score": avg_prev,
                "fake_prob": 0.5,
                "fake_prob_smooth": avg_prev,
                "waveform": np.zeros(120, dtype=np.float32),
                "bpm": 0.0,
                "snr": -np.inf,
                "snr_12hz_ratio": 0.0,
                "is_bpm_reliable": False,
                "fft_thumb": np.zeros((64, 64, 3), dtype=np.uint8),
                "status": "Scanning...",
                "scan_pulse": pulse,
            }

        if self._tracker_mode != "tasks_detector":
            cx, cy = w // 2, h // 2
            fw, fh = int(w * 0.34), int(h * 0.52)
            x0, y0, x1, y1 = _clamp_bbox(cx - fw // 2, cy - fh // 2, cx + fw // 2, cy + fh // 2, w, h)
            face = frame_bgr[y0:y1, x0:x1]
            if face.size == 0:
                pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 2.4))
                return _idle(pulse)

            fh2, fw2 = face.shape[:2]
            r_top = face[int(fh2 * 0.10) : int(fh2 * 0.34), int(fw2 * 0.25) : int(fw2 * 0.75)]
            r_l = face[int(fh2 * 0.40) : int(fh2 * 0.78), int(fw2 * 0.08) : int(fw2 * 0.42)]
            r_r = face[int(fh2 * 0.40) : int(fh2 * 0.78), int(fw2 * 0.58) : int(fw2 * 0.92)]
            rois = [r for r in [r_top, r_l, r_r] if r.size > 0]
            if not rois:
                rgb_mean = np.zeros(3, dtype=np.float32)
            else:
                means = []
                for r in rois:
                    bgr = np.mean(r.reshape(-1, 3), axis=0)
                    means.append(np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32))
                rgb_mean = np.mean(np.stack(means, axis=0), axis=0)
        else:
            bbox = self._detect_bbox_tasks(frame_bgr)
            if bbox is None:
                pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 2.4))
                return _idle(pulse)
            x0, y0, x1, y1 = bbox
            face = frame_bgr[y0:y1, x0:x1]
            fh2, fw2 = face.shape[:2]
            r_top = face[int(fh2 * 0.10) : int(fh2 * 0.34), int(fw2 * 0.25) : int(fw2 * 0.75)]
            r_l = face[int(fh2 * 0.40) : int(fh2 * 0.78), int(fw2 * 0.08) : int(fw2 * 0.42)]
            r_r = face[int(fh2 * 0.40) : int(fh2 * 0.78), int(fw2 * 0.58) : int(fw2 * 0.92)]
            rois = [r for r in [r_top, r_l, r_r] if r.size > 0]
            if not rois:
                rgb_mean = np.zeros(3, dtype=np.float32)
            else:
                means = []
                for r in rois:
                    bgr = np.mean(r.reshape(-1, 3), axis=0)
                    means.append(np.array([bgr[2], bgr[1], bgr[0]], dtype=np.float32))
                rgb_mean = np.mean(np.stack(means, axis=0), axis=0)

        if face.size == 0:
            pulse = 0.55 + 0.45 * (0.5 + 0.5 * math.sin(time.time() * 2.4))
            return _idle(pulse)

        self._rgb_trace.append(rgb_mean)

        lap_var = _laplacian_var_bgr(face)
        low_quality = lap_var < LAPLACIAN_VAR_THRESHOLD
        face_u = _unsharp_mask_bgr(_mcdm_preprocess_placeholder(face))
        rgb_t = _to_model_tensor_rgb(face_u).to(self._device)
        fft_t, fft_thumb_full = _to_model_tensor_fft(face_u)
        fft_t = fft_t.to(self._device)
        with torch.no_grad():
            fake_prob = float(
                self._model(rgb_t, fft_t, force_equal_dfa=low_quality).item()
            )

        calib = _calibrate_fake_score(fake_prob)
        self._calib_hist.append(calib)
        avg_fake = float(np.mean(self._calib_hist))

        rgb_trace = np.asarray(self._rgb_trace, dtype=np.float32)
        pos = _pos_signal(rgb_trace, fps=fs_sample)
        pos_f = _filter_pos(pos, fps=fs_sample)
        bpm, snr_db, snr_12, is_bpm_ok = _estimate_bpm_and_snr_metrics(pos_f, fs_sample)
        waveform = pos_f[-120:] if len(pos_f) > 0 else np.zeros(120, dtype=np.float32)
        if len(waveform) < 120:
            waveform = np.pad(waveform, (120 - len(waveform), 0))

        status = "Safe"
        if avg_fake >= 0.65:
            status = "Purifying"
        elif avg_fake >= 0.45:
            status = "Caution"

        return {
            "face_found": True,
            "bbox": (x0, y0, x1, y1),
            "raw_fake_score": fake_prob,
            "calib_fake_score": calib,
            "avg_fake_score": avg_fake,
            "fake_prob": fake_prob,
            "fake_prob_smooth": avg_fake,
            "waveform": waveform.astype(np.float32),
            "bpm": bpm,
            "snr": snr_db,
            "snr_12hz_ratio": snr_12,
            "is_bpm_reliable": is_bpm_ok,
            "fft_thumb": cv2.resize(fft_thumb_full, (64, 64), interpolation=cv2.INTER_AREA),
            "status": status,
            "scan_pulse": 1.0,
        }


def _draw_glass_panel(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, alpha: float = 0.72) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 32, 36), -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)
    cv2.rectangle(img, (x0, y0), (x1, y1), SOFT_WHITE, 1)


def _draw_ring_progress(img: np.ndarray, center: tuple[int, int], radius: int, score: float) -> None:
    score = float(np.clip(score, 0.0, 1.0))
    safe = np.array(PASTEL_BLUE, dtype=np.float32)
    danger = np.array(SOFT_PINK, dtype=np.float32)
    color_arr = safe * (1.0 - score) + danger * score
    color = tuple(int(c) for c in color_arr)
    cv2.circle(img, center, radius, (70, 72, 78), 10)
    end_angle = int(360 * score)
    cv2.ellipse(img, center, (radius, radius), -90, 0, end_angle, color, 10)
    cv2.putText(img, f"{score:.2f}", (center[0] - 28, center[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BEIGE, 2)


def _draw_waveform(img: np.ndarray, wave: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    _draw_glass_panel(img, x0, y0, x1, y1, alpha=0.66)
    if len(wave) < 2:
        return
    w = x1 - x0 - 12
    h = y1 - y0 - 18
    pts = []
    wv = wave.copy()
    wv = (wv - np.min(wv)) / (np.max(wv) - np.min(wv) + 1e-6)
    for i, v in enumerate(wv):
        px = x0 + 6 + int(i * w / max(len(wv) - 1, 1))
        py = y0 + 8 + int((1.0 - v) * h)
        pts.append((px, py))
    cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False, PASTEL_BLUE, 2)
    cv2.putText(img, "Live rPPG", (x0 + 8, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BEIGE, 1)


def draw_console(
    fps: float,
    latency_ms: float,
    stability: float,
    bpm: float,
    fake_score: float,
    raw_fake_score: float,
    status: str,
    waveform: np.ndarray,
    fft_thumb: np.ndarray,
    face_found: bool,
    scan_pulse: float,
    is_bpm_reliable: bool,
) -> np.ndarray:
    panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    _draw_glass_panel(panel, 6, 6, PANEL_WIDTH - 7, PANEL_HEIGHT - 7, alpha=0.72)

    cv2.putText(panel, "PREVERVECT", (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.95, BEIGE, 2)
    cv2.putText(panel, "Research Defense Dashboard", (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, SOFT_WHITE, 1)

    _draw_ring_progress(panel, (92, 150), 46, fake_score)
    cv2.putText(panel, "Fake Score (avg)", (28, 218), cv2.FONT_HERSHEY_SIMPLEX, 0.48, SOFT_WHITE, 1)
    cv2.putText(
        panel,
        f"raw {raw_fake_score:.2f}",
        (28, 236),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        PASTEL_BLUE,
        1,
    )

    thumb_rect = (PANEL_WIDTH - 96, 120, PANEL_WIDTH - 32, 184)
    _draw_glass_panel(panel, thumb_rect[0] - 4, thumb_rect[1] - 4, thumb_rect[2] + 4, thumb_rect[3] + 4, alpha=0.60)
    panel[thumb_rect[1] : thumb_rect[3], thumb_rect[0] : thumb_rect[2]] = fft_thumb
    cv2.putText(panel, "FFT", (thumb_rect[0] + 16, thumb_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.48, SOFT_WHITE, 1)

    _draw_waveform(panel, waveform, (24, 242, PANEL_WIDTH - 24, 332))

    gx0, gy0 = 24, 352
    gx1, gy1 = PANEL_WIDTH - 24, 470
    _draw_glass_panel(panel, gx0, gy0, gx1, gy1, alpha=0.66)
    midx = (gx0 + gx1) // 2
    midy = (gy0 + gy1) // 2
    cv2.line(panel, (midx, gy0), (midx, gy1), (120, 125, 132), 1)
    cv2.line(panel, (gx0, midy), (gx1, midy), (120, 125, 132), 1)
    cv2.putText(panel, "FPS", (gx0 + 16, gy0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, SOFT_WHITE, 1)
    cv2.putText(panel, f"{fps:4.1f}", (gx0 + 16, gy0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BEIGE, 1)

    cv2.putText(panel, "Latency", (midx + 12, gy0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.54, SOFT_WHITE, 1)
    cv2.putText(panel, f"{latency_ms:5.1f} ms", (midx + 12, gy0 + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BEIGE, 1)

    cv2.putText(panel, "Stability", (gx0 + 16, midy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, SOFT_WHITE, 1)
    cv2.putText(panel, f"{stability:4.2f}", (gx0 + 16, midy + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BEIGE, 1)

    cv2.putText(panel, "Heart Rate", (midx + 12, midy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.54, SOFT_WHITE, 1)
    if is_bpm_reliable and face_found:
        cv2.putText(panel, f"{bpm:5.1f} BPM", (midx + 12, midy + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BEIGE, 1)
    else:
        cv2.putText(panel, "偵測中", (midx + 12, midy + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BEIGE, 1)

    pulse = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(time.time() * 2.5))
    if status == "Purifying":
        glow = tuple(int(c * pulse) for c in SOFT_PINK)
    elif not face_found:
        glow = tuple(int(c * scan_pulse) for c in PASTEL_BLUE)
    else:
        glow = SOFT_WHITE
    cv2.putText(panel, f"Status: {status}", (24, 510), cv2.FONT_HERSHEY_SIMPLEX, 0.68, glow, 2)
    cv2.putText(panel, "Press q to quit", (24, 536), cv2.FONT_HERSHEY_SIMPLEX, 0.52, SOFT_WHITE, 1)
    return panel


def _to_live_thumbnail(frame: np.ndarray, max_width: int = 760, max_height: int = 420) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
    return cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


def _make_session_logger(log_dir: Path) -> tuple[csv.DictWriter, object]:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"session_{ts}.csv"
    f = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "time_sec",
            "fps",
            "latency_ms",
            "bpm",
            "bpm_reliable",
            "snr_db",
            "snr_12hz_ratio",
            "raw_fake_score",
            "avg_fake_score",
        ],
    )
    writer.writeheader()
    print(f"[INFO] Session log: {path}")
    return writer, f


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    worker = InferenceWorker(device=device, weights_path=args.weights)
    worker.start()
    logger, log_fp = _make_session_logger(args.log_dir)

    with mss.mss() as sct:
        monitor = _get_monitor_from_window_title(args.target_window)
        if monitor is None:
            monitor = {"top": 120, "left": 160, "width": 1280, "height": 720}

        cv2.namedWindow(LIVE_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(CONSOLE_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(LIVE_WINDOW_NAME, 760, 420)
        cv2.resizeWindow(CONSOLE_WINDOW_NAME, PANEL_WIDTH, PANEL_HEIGHT)
        cv2.moveWindow(LIVE_WINDOW_NAME, 30, 40)
        cv2.moveWindow(CONSOLE_WINDOW_NAME, 840, 40)
        try:
            cv2.setWindowProperty(CONSOLE_WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
        except Exception:
            pass

        fps_hist: deque[float] = deque(maxlen=40)
        prev = time.perf_counter()
        last_log_t = 0.0
        while True:
            shot = sct.grab(monitor)
            frame = cv2.cvtColor(np.array(shot, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
            worker.update_frame(frame)
            result = worker.get_result()

            now = time.perf_counter()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            fps_hist.append(fps)
            fps_stability = float(np.std(np.asarray(fps_hist))) if len(fps_hist) > 2 else 0.0

            if result["face_found"] and result["bbox"] is not None:
                x0, y0, x1, y1 = result["bbox"]
                color = SOFT_PINK if float(result.get("avg_fake_score", result["fake_prob_smooth"])) >= 0.5 else SOFT_WHITE
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            else:
                pulse = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(time.time() * 2.4))
                txt_color = tuple(int(c * pulse) for c in PASTEL_BLUE)
                cv2.putText(frame, "Scanning...", (30, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0, txt_color, 2)

            live_thumb = _to_live_thumbnail(frame)
            console = draw_console(
                fps=fps,
                latency_ms=float(result["latency_ms"]),
                stability=fps_stability,
                bpm=float(result["bpm"]),
                fake_score=float(result.get("avg_fake_score", result["fake_prob_smooth"])),
                raw_fake_score=float(result.get("raw_fake_score", result["fake_prob"])),
                status=str(result["status"]),
                waveform=result["waveform"],
                fft_thumb=result["fft_thumb"],
                face_found=bool(result["face_found"]),
                scan_pulse=float(result.get("scan_pulse", 1.0)),
                is_bpm_reliable=bool(result.get("is_bpm_reliable", False)),
            )
            cv2.imshow(LIVE_WINDOW_NAME, live_thumb)
            cv2.imshow(CONSOLE_WINDOW_NAME, console)

            t = time.time()
            if t - last_log_t >= 1.0:
                logger.writerow(
                    {
                        "time_sec": f"{t:.3f}",
                        "fps": f"{fps:.3f}",
                        "latency_ms": f"{float(result['latency_ms']):.3f}",
                        "bpm": f"{float(result['bpm']):.3f}",
                        "bpm_reliable": str(bool(result.get("is_bpm_reliable", False))),
                        "snr_db": f"{float(result['snr']):.3f}",
                        "snr_12hz_ratio": f"{float(result.get('snr_12hz_ratio', 0.0)):.4f}",
                        "raw_fake_score": f"{float(result.get('raw_fake_score', result['fake_prob'])):.6f}",
                        "avg_fake_score": f"{float(result.get('avg_fake_score', result['fake_prob_smooth'])):.6f}",
                    }
                )
                log_fp.flush()
                last_log_t = t

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    worker.stop()
    worker.join(timeout=1.0)
    log_fp.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
