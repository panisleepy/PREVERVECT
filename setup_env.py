"""Environment setup script for PREVERVECT."""

from __future__ import annotations

from pathlib import Path

import timm
import torch


def save_efficientnet_b0_weights(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = timm.create_model("efficientnet_b0", pretrained=True)
    torch.save(model.state_dict(), output_path)
    print(f"[OK] Saved pretrained weights to: {output_path}")


def test_mediapipe_import() -> None:
    import mediapipe as mp  # noqa: F401

    print("[OK] mediapipe import test passed")


def test_mss_capture() -> None:
    import mss
    import numpy as np

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        frame = np.array(shot)
        if frame.size == 0:
            raise RuntimeError("mss capture returned empty frame")
    print("[OK] mss capture test passed")


def main() -> None:
    weight_file = Path("weights/efficientnet_b0_pretrained.pth")
    save_efficientnet_b0_weights(weight_file)
    test_mediapipe_import()
    test_mss_capture()
    print("[DONE] Environment bootstrap completed")


if __name__ == "__main__":
    main()
