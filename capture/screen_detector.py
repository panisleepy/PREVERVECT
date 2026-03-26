"""Realtime screen capture + face ROI + SpecXNet inference."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import mss
import numpy as np
import torch

# Ensure project-root imports work when running:
#   python capture/screen_detector.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.model import build_specxnet
from utils.fft_tools import bgr_to_tensor, fft_to_tensor


WINDOW_NAME = "PREVERVECT Realtime Detector"
PANEL_WIDTH = 300


def _create_face_detector():
    """
    Create face detector with robust fallback.
    1) MediaPipe FaceMesh (if mp.solutions exists in current package build)
    2) OpenCV Haar cascade fallback
    """
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return "mediapipe", face_mesh

    # Fallback for environments where `mediapipe.solutions` is not exposed.
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError("Failed to load OpenCV Haar cascade face detector.")
    return "opencv", cascade


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime screen deepfake detector")
    parser.add_argument(
        "--target_window",
        type=str,
        default="",
        help='Window title keyword for auto capture (e.g. "Chrome", "YouTube").',
    )
    return parser.parse_args()


def _get_monitor_from_window_title(title_keyword: str) -> dict[str, int] | None:
    """
    Try to locate an OS window by title and return its rectangle as monitor.
    Returns None if pygetwindow is unavailable or no match is found.
    """
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

    # Pick the largest matching visible window.
    best = max(candidates, key=lambda w: max(0, w.width) * max(0, w.height))
    left, top, width, height = int(best.left), int(best.top), int(best.width), int(best.height)
    if width <= 0 or height <= 0:
        return None
    return {"top": top, "left": left, "width": width, "height": height}


def _mcdm_preprocess_placeholder(face_roi: np.ndarray) -> np.ndarray:
    """
    Placeholder for future MCDM defense pipeline.
    Keep this function in the inference path as an integration point.
    """
    return face_roi


def _detect_face_bbox(frame_bgr: np.ndarray, detector_type: str, detector_obj):
    """Return a single face bbox (x0, y0, x1, y1) or None."""
    h, w = frame_bgr.shape[:2]
    if detector_type == "mediapipe":
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = detector_obj.process(rgb)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0].landmark
        xs = np.array([lm.x for lm in landmarks]) * w
        ys = np.array([lm.y for lm in landmarks]) * h
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max()), int(ys.max())
    else:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = detector_obj.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        # pick largest face
        x, y, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        x0, y0, x1, y1 = int(x), int(y), int(x + fw), int(y + fh)

    pad_x = int((x1 - x0) * 0.15)
    pad_y = int((y1 - y0) * 0.20)
    return clamp_bbox(x0 - pad_x, y0 - pad_y, x1 + pad_x, y1 + pad_y, w, h)


def load_model(device: torch.device) -> torch.nn.Module:
    """
    Load SpecXNet and optional fine-tuned weights.
    If no fine-tuned file exists, pretrained backbone weights are used for flow testing.
    """
    model = build_specxnet(pretrained=True, device=device)
    weight_path = Path("weights/specxnet_finetuned.pt")
    if weight_path.exists():
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def clamp_bbox(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, min(x0, w - 1))
    y0 = max(0, min(y0, h - 1))
    x1 = max(1, min(x1, w))
    y1 = max(1, min(y1, h))
    return x0, y0, x1, y1


def _draw_capture_guide(frame: np.ndarray) -> tuple[int, int, int, int]:
    """Draw where users should place video content in the capture preview."""
    h, w = frame.shape[:2]
    gw = int(w * 0.66)
    gh = int(h * 0.82)
    x0 = (w - gw) // 2
    y0 = (h - gh) // 2
    x1 = x0 + gw
    y1 = y0 + gh
    color = (120, 220, 255)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    cv2.putText(
        frame,
        "Detection Range (place video inside)",
        (x0 + 10, max(25, y0 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    return x0, y0, x1, y1


def _render_overlay_panel(
    frame: np.ndarray,
    trust_score: float,
    fps: float,
    status: str,
    detector_type: str,
) -> np.ndarray:
    """Render right-side panel and return final composited frame."""
    h, _ = frame.shape[:2]
    panel = np.zeros((h, PANEL_WIDTH, 3), dtype=np.uint8)
    panel[:] = (28, 26, 24)

    # Semi-transparent style.
    blended = cv2.addWeighted(panel, 0.68, np.full_like(panel, (52, 46, 42)), 0.32, 0.0)

    text_primary = (210, 230, 240)
    text_soft = (190, 205, 215)
    bar_bg = (70, 74, 82)
    bar_fg = (120, 210, 255)

    cv2.putText(blended, "PREVERVECT", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_primary, 2)
    cv2.putText(blended, "Realtime Defense Console", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_soft, 1)

    cv2.putText(blended, "Trust Score", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_primary, 1)
    bx0, by0, bx1, by1 = 20, 125, PANEL_WIDTH - 20, 150
    cv2.rectangle(blended, (bx0, by0), (bx1, by1), bar_bg, -1)
    fill_w = int((bx1 - bx0) * float(np.clip(trust_score, 0.0, 1.0)))
    cv2.rectangle(blended, (bx0, by0), (bx0 + fill_w, by1), bar_fg, -1)
    cv2.putText(blended, f"{trust_score:.3f}", (bx1 - 90, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_primary, 1)

    cv2.putText(blended, f"FPS: {fps:5.1f}", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_primary, 1)
    cv2.putText(blended, f"Status: {status}", (20, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_primary, 1)
    cv2.putText(blended, f"Detector: {detector_type}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_soft, 1)
    cv2.putText(blended, "Press q to quit", (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, text_soft, 1)

    return np.hstack([frame, blended])


def main() -> None:
    args = _parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    detector_type, detector_obj = _create_face_detector()

    with mss.mss() as sct:
        monitor = _get_monitor_from_window_title(args.target_window)
        if monitor is None:
            # fallback to previous manual region
            monitor = {"top": 120, "left": 160, "width": 1280, "height": 720}

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        # Move preview window away from capture region to avoid recursion.
        display_x = monitor["left"] + monitor["width"] + 30
        display_y = max(40, monitor["top"])
        cv2.moveWindow(WINDOW_NAME, display_x, display_y)

        prev = time.time()
        while True:
            shot = sct.grab(monitor)
            # Convert via OpenCV to keep a contiguous/writeable array for drawing APIs.
            frame = cv2.cvtColor(np.array(shot, dtype=np.uint8), cv2.COLOR_BGRA2BGR)
            _draw_capture_guide(frame)
            bbox = _detect_face_bbox(frame, detector_type, detector_obj)

            trust_score = 0.5
            status = "Analyzing"

            if bbox is not None:
                x0, y0, x1, y1 = bbox
                face_roi = frame[y0:y1, x0:x1]
                if face_roi.size > 0:
                    face_roi_mcdm = _mcdm_preprocess_placeholder(face_roi)
                    rgb_tensor = bgr_to_tensor(face_roi_mcdm).to(device)
                    fft_tensor = fft_to_tensor(face_roi_mcdm).to(device)
                    with torch.no_grad():
                        fake_prob = model(rgb_tensor, fft_tensor).item()

                    # Define trust score as "realness" for intuitive UI.
                    trust_score = float(1.0 - fake_prob)
                    if trust_score < 0.40:
                        status = "Warning"
                    elif trust_score < 0.65:
                        status = "Purifying"
                    else:
                        status = "Analyzing"

                    color = (80, 220, 140) if trust_score >= 0.5 else (40, 80, 240)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
            else:
                status = "Analyzing"

            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now

            composed = _render_overlay_panel(frame, trust_score, fps, status, detector_type)
            cv2.imshow(WINDOW_NAME, composed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if detector_type == "mediapipe":
        detector_obj.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
