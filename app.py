"""FastAPI 微服務：PREVERVECT Chrome 外掛與 SpecXNet 推論橋接。"""

from __future__ import annotations

import base64
import logging
import urllib.request
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any
import shutil

import cv2
import mediapipe as mp
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scipy.signal import butter, filtfilt, welch

from core.model import build_specxnet
from utils.fft_tools import power_spectrum_shifted_bgr

# ---------------------------------------------------------------------------
# 常數：與 core/dataloader、capture/screen_detector 對齊
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
TMP_MODEL_DIR = Path(r"C:\temp\prevervect_models")
MODEL_SIZE = 224
BPM_SNR_12HZ_THRESHOLD = 1.5
# 拉普拉斯變異數過低時視為低畫質，強制 DFA 雙流各 50% 融合
LAPLACIAN_VAR_THRESHOLD = 120.0

# FakeScore：校準 + Moving Average + rPPG reliability 保守回拉（與 capture/screen_detector.py 對齊）
SCORE_CALIB_OFFSET = 0.3
SCORE_CALIB_DIV = 0.7
SCORE_SMOOTH_LEN = 15
SCORE_RPPG_W_MIN = 0.35
RPPG_SNR_W_MAX = 3.0

app = FastAPI(title="PREVERVECT API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectRequest(BaseModel):
    """前端 POST 本文：含 Base64 或 Data URL 圖片字串。"""

    image_base64: str


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    """將邊界框限制在影像範圍內。"""
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(x0 + 1, min(x1, w))
    y1 = max(y0 + 1, min(y1, h))
    return x0, y0, x1, y1


def _expand_bbox_20pct(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    """以中心為基準將框各向擴張約 20%（邊長 ×1.2），保留邊緣與背景上下文。"""
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = float(x1 - x0)
    bh = float(y1 - y0)
    nw = bw * 1.2
    nh = bh * 1.2
    x0n = int(round(cx - nw * 0.5))
    y0n = int(round(cy - nh * 0.5))
    x1n = int(round(cx + nw * 0.5))
    y1n = int(round(cy + nh * 0.5))
    return _clamp_bbox(x0n, y0n, x1n, y1n, w, h)


def _crop_square_roi(frame_bgr: np.ndarray, x0: int, y0: int, x1: int, y1: int) -> np.ndarray | None:
    """
    在擴張後的矩形內裁切**正方形**臉部 ROI（邊長 = max(寬, 高)，置中），並夾在圖內。
    若可用區域過小則回傳 None。
    """
    h, w = frame_bgr.shape[:2]
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = int(max(x1 - x0, y1 - y0))
    if side < 8:
        return None
    half = side * 0.5
    xs0 = int(round(cx - half))
    ys0 = int(round(cy - half))
    xs1 = xs0 + side
    ys1 = ys0 + side
    if xs0 < 0:
        xs1 -= xs0
        xs0 = 0
    if ys0 < 0:
        ys1 -= ys0
        ys0 = 0
    if xs1 > w:
        d = xs1 - w
        xs0 -= d
        xs1 = w
    if ys1 > h:
        d = ys1 - h
        ys0 -= d
        ys1 = h
    xs0 = max(0, xs0)
    ys0 = max(0, ys0)
    xs1 = min(w, xs1)
    ys1 = min(h, ys1)
    sw = xs1 - xs0
    sh = ys1 - ys0
    side2 = min(sw, sh)
    if side2 < 8:
        return None
    ox = (sw - side2) // 2
    oy = (sh - side2) // 2
    return frame_bgr[ys0 + oy : ys0 + oy + side2, xs0 + ox : xs0 + ox + side2].copy()


def _laplacian_var_bgr(bgr: np.ndarray) -> float:
    """用拉普拉斯變異數估計對焦／清晰度。"""
    if bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _ensure_blaze_face_model() -> Path | None:
    """下載並快取 MediaPipe BlazeFace 短距離模型（與 screen_detector 一致）。"""
    model_path = WEIGHTS_DIR / "blaze_face_short_range.tflite"
    if model_path.is_file():
        return _copy_to_ascii_tmp(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )
    try:
        print(f"[INFO] 下載 MediaPipe 人臉偵測模型: {url}")
        urllib.request.urlretrieve(url, str(model_path))
        return _copy_to_ascii_tmp(model_path)
    except Exception as e:
        print(f"[WARN] BlazeFace 下載失敗: {e}")
        return None


def _copy_to_ascii_tmp(src: Path, dst_name: str | None = None) -> Path:
    """
    MediaPipe tasks 在 Windows 上對 Unicode 路徑支援不穩，會出現 errno=-1。
    將模型複製到 ASCII-only 的 temp 路徑後再載入最穩。
    """
    TMP_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dst = TMP_MODEL_DIR / (dst_name or src.name)
    # Copy overwrites if exists.
    shutil.copyfile(str(src), str(dst))
    return dst


class DetectorService:
    """載入 SpecXNet、人臉偵測器，處理 /detect 單張推論。"""

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_specxnet(pretrained=True, device=self.device)
        self.model.eval()
        self._load_weights()

        # MediaPipe Tasks Face Detector（優先）；失敗則 Haar
        self._mp_face_detector: Any = None
        self._haar: cv2.CascadeClassifier | None = None
        self._init_face_detectors()

        # rPPG 輕量時序（臉部 ROI 平均 RGB），用於 is_reliable
        self.rgb_trace: deque[np.ndarray] = deque(maxlen=240)
        self.dt_trace: deque[float] = deque(maxlen=90)
        self._last_t: float | None = None

        # FakeScore：calibration 後的 deque(15) 移動平均（與桌面版一致）
        self._calib_score_hist: deque[float] = deque(maxlen=SCORE_SMOOTH_LEN)

    @staticmethod
    def _calibrate_fake_score(raw: float) -> float:
        x = (float(raw) - SCORE_CALIB_OFFSET) / SCORE_CALIB_DIV
        return float(np.clip(x, 0.0, 1.0))

    def _init_face_detectors(self) -> None:
        """初始化 MediaPipe Face Detection；不可用則回退 OpenCV Haar。"""
        try:
            if hasattr(mp, "tasks"):
                model_path = _ensure_blaze_face_model()
                if model_path is not None:
                    BaseOptions = mp.tasks.BaseOptions
                    vision = mp.tasks.vision
                    options = vision.FaceDetectorOptions(
                        base_options=BaseOptions(model_asset_path=str(model_path.resolve())),
                        # 降低門檻以提升在網頁 1FPS / 壓縮畫面下的可偵測性
                        min_detection_confidence=0.3,
                        running_mode=vision.RunningMode.IMAGE,
                    )
                    self._mp_face_detector = vision.FaceDetector.create_from_options(options)
                    print("[INFO] MediaPipe Face Detector（BlazeFace）已就緒。")
                    return
        except Exception as e:
            print(f"[WARN] MediaPipe Face Detector 初始化失敗: {e}")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        # 同樣處理 unicode 路徑風險：複製到 ASCII-only temp 再載入
        try:
            cascade_src = Path(cascade_path)
            if cascade_src.is_file():
                cascade_tmp = _copy_to_ascii_tmp(
                    cascade_src,
                    dst_name="haarcascade_frontalface_default.xml",
                )
                self._haar = cv2.CascadeClassifier(str(cascade_tmp))
            else:
                self._haar = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self._haar = cv2.CascadeClassifier(cascade_path)
        if self._haar.empty():
            self._haar = None
            print("[ERROR] Haar 分類器無法載入。")
        else:
            print("[INFO] 使用 OpenCV Haar Cascade 作為人臉偵測備案。")

    def _load_weights(self) -> None:
        """依序載入微調權重；無則僅 ImageNet 預訓練骨干。"""
        candidates = [
            WEIGHTS_DIR / "specxnet_best.pth",
            WEIGHTS_DIR / "specxnet_latest.pth",
            WEIGHTS_DIR / "specxnet_finetuned.pt",
        ]
        for p in candidates:
            if p.is_file():
                try:
                    state = torch.load(p, map_location=self.device, weights_only=True)
                except TypeError:
                    state = torch.load(p, map_location=self.device)
                self.model.load_state_dict(state, strict=False)
                print(f"[INFO] 已載入 SpecXNet 權重: {p.resolve()}")
                return
        print("[WARN] 未找到微調權重，分類頭僅為隨機初始化＋預訓練骨干。")

    @staticmethod
    def _decode_base64(data_url: str) -> np.ndarray:
        """將 Base64 / Data URL 解碼為 BGR uint8。"""
        try:
            payload = data_url.split(",", 1)[1] if "," in data_url else data_url
            raw = base64.b64decode(payload)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc
        if frame is None:
            raise HTTPException(status_code=400, detail="Cannot decode image data.")
        return frame

    def _detect_largest_face_bbox(self, frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
        """回傳畫面中信心分數最高（MediaPipe）或面積最大（Haar）的人臉框。"""
        h, w = frame_bgr.shape[:2]
        if self._mp_face_detector is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            det_result = self._mp_face_detector.detect(mp_image)
            if not det_result.detections:
                return None
            best = max(
                det_result.detections,
                key=lambda d: d.categories[0].score if d.categories else 0.0,
            )
            bb = best.bounding_box
            x0 = int(bb.origin_x)
            y0 = int(bb.origin_y)
            x1 = int(bb.origin_x + bb.width)
            y1 = int(bb.origin_y + bb.height)
            return _clamp_bbox(x0, y0, x1, y1, w, h)

        if self._haar is not None:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            # 網頁影片縮圖/縮放後人臉可能偏小，降低 minSize 提升召回
            faces = self._haar.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=3,
                minSize=(32, 32),
            )
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
            return _clamp_bbox(int(x), int(y), int(x + fw), int(y + fh), w, h)
        return None

    @staticmethod
    def _to_tensor_rgb(face_bgr: np.ndarray) -> torch.Tensor:
        """224×224、ImageNet 正規化，與訓練 DataLoader 一致。"""
        fh, fw = face_bgr.shape[:2]
        interp = cv2.INTER_AREA if fh >= MODEL_SIZE and fw >= MODEL_SIZE else cv2.INTER_LINEAR
        resized = cv2.resize(face_bgr, (MODEL_SIZE, MODEL_SIZE), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std
        return torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)

    @staticmethod
    def _to_tensor_fft(face_bgr: np.ndarray) -> torch.Tensor:
        """頻域功率譜 → 224×224 → [-1,1] 正規化。"""
        spec = power_spectrum_shifted_bgr(face_bgr)
        sh, sw = spec.shape[:2]
        interp = cv2.INTER_AREA if sh >= MODEL_SIZE and sw >= MODEL_SIZE else cv2.INTER_LINEAR
        spec_resized = cv2.resize(spec, (MODEL_SIZE, MODEL_SIZE), interpolation=interp).astype(np.float32)
        norm = (spec_resized - 0.5) / 0.5
        return torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)

    @staticmethod
    def _filter_pos(sig: np.ndarray, fps: float) -> np.ndarray:
        """心率帶通（約 0.7–3 Hz），供 Welch 估 SNR。"""
        if len(sig) < 8:
            return sig.astype(np.float32)
        nyq = 0.5 * fps
        low = 0.7 / nyq
        high = 3.0 / nyq
        low = max(low, 1e-4)
        high = min(high, 0.999)
        if not (0 < low < high < 1):
            return sig.astype(np.float32)
        b, a = butter(2, [low, high], btype="bandpass")
        # filtfilt 要求 len(sig) > padlen，否則 ValueError（常見於 1 FPS 前幾幀時序仍短）
        padlen = 3 * (max(len(a), len(b)) - 1)
        if len(sig) <= padlen:
            return sig.astype(np.float32)
        return filtfilt(b, a, sig).astype(np.float32)

    @staticmethod
    def _estimate_bpm_reliable(sig: np.ndarray, fs: float) -> tuple[bool, float]:
        """回傳 (is_reliable, snr_12hz_ratio)。"""
        if len(sig) < 16 or fs <= 0:
            return False, 0.0
        nperseg = min(256, len(sig))
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        band = (freqs >= 0.5) & (freqs <= 4.0)
        if not np.any(band):
            return False, 0.0
        p_full = float(np.sum(psd[band])) + 1e-12
        band_12 = (freqs >= 1.1) & (freqs <= 1.3)
        p_12 = float(np.sum(psd[band_12]))
        rest = max(p_full - p_12, 1e-12)
        snr_12hz_ratio = p_12 / rest
        return bool(snr_12hz_ratio >= BPM_SNR_12HZ_THRESHOLD), float(snr_12hz_ratio)

    def detect(self, image_base64: str) -> dict[str, Any]:
        """
        /detect 核心邏輯：
        解碼 → 人臉 → 擴張 20% → 正方形裁切 → 雙流推論。
        無臉時不跑模型，僅回傳約定 JSON。
        """
        frame = self._decode_base64(image_base64)
        H, W = frame.shape[:2]

        bbox = self._detect_largest_face_bbox(frame)
        if bbox is None:
            return {
                "fake_score": 0,
                "raw_fake_score": 0.0,
                "calib_fake_score": 0.0,
                "avg_fake_score": 0.0,
                "snr_12hz_ratio": 0.0,
                "is_reliable": False,
                "message": "No face detected",
            }

        x0, y0, x1, y1 = bbox
        xe0, ye0, xe1, ye1 = _expand_bbox_20pct(x0, y0, x1, y1, W, H)
        face_roi = _crop_square_roi(frame, xe0, ye0, xe1, ye1)
        if face_roi is None or face_roi.size == 0:
            return {
                "fake_score": 0,
                "raw_fake_score": 0.0,
                "calib_fake_score": 0.0,
                "avg_fake_score": 0.0,
                "snr_12hz_ratio": 0.0,
                "is_reliable": False,
                "message": "No face detected",
            }

        rgb_mean = np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB).reshape(-1, 3), axis=0).astype(np.float32)
        self.rgb_trace.append(rgb_mean)
        now = perf_counter()
        if self._last_t is not None:
            dt = now - self._last_t
            if dt > 1e-4:
                self.dt_trace.append(dt)
        self._last_t = now
        fs = 30.0 if len(self.dt_trace) < 5 else float(np.clip(1.0 / max(np.mean(self.dt_trace), 1e-3), 8.0, 120.0))

        lap_var = _laplacian_var_bgr(face_roi)
        low_quality = lap_var < LAPLACIAN_VAR_THRESHOLD

        rgb_t = self._to_tensor_rgb(face_roi).to(self.device)
        fft_t = self._to_tensor_fft(face_roi).to(self.device)
        with torch.no_grad():
            fake_score = float(self.model(rgb_t, fft_t, force_equal_dfa=low_quality).item())

        # FakeScore：校準 + deque(15) 移動平均（img trust）
        calib_fake_score = self._calibrate_fake_score(fake_score)
        self._calib_score_hist.append(calib_fake_score)
        avg_fake_score_img = float(np.mean(self._calib_score_hist))

        trace = np.stack(self.rgb_trace, axis=0) if self.rgb_trace else np.zeros((1, 3), dtype=np.float32)
        sig = trace[:, 1] - trace[:, 0]
        sig = sig - np.mean(sig)
        if np.std(sig) > 1e-6:
            sig = sig / (np.std(sig) + 1e-6)
        sig_f = self._filter_pos(sig, fps=fs)
        is_reliable, snr_12hz_ratio = self._estimate_bpm_reliable(sig_f, fs=fs)

        # rPPG reliability → conservative回拉
        denom = max(RPPG_SNR_W_MAX - BPM_SNR_12HZ_THRESHOLD, 1e-6)
        w_raw = float(np.clip((snr_12hz_ratio - BPM_SNR_12HZ_THRESHOLD) / denom, 0.0, 1.0))
        w = float(SCORE_RPPG_W_MIN + (1.0 - SCORE_RPPG_W_MIN) * w_raw)
        avg_fake_score = 0.5 + w * (avg_fake_score_img - 0.5)

        # 兼容舊前端：保留 fake_score 欄位（回傳 avg 版本，讓視覺更一致）
        return {
            "fake_score": avg_fake_score,
            "raw_fake_score": fake_score,
            "calib_fake_score": calib_fake_score,
            "avg_fake_score": avg_fake_score,
            "snr_12hz_ratio": snr_12hz_ratio,
            "is_reliable": bool(is_reliable),
        }


service = DetectorService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detect")
def detect(req: DetectRequest) -> dict[str, Any]:
    """接收 Base64 圖片，回傳 fake_score、is_reliable；無臉時含 message。"""
    try:
        return service.detect(req.image_base64)
    except HTTPException:
        raise
    except Exception:
        logging.exception("POST /detect 處理失敗")
        return {
            "fake_score": 0.0,
            "raw_fake_score": 0.0,
            "calib_fake_score": 0.0,
            "avg_fake_score": 0.0,
            "snr_12hz_ratio": 0.0,
            "is_reliable": False,
            "message": "Inference error",
        }
