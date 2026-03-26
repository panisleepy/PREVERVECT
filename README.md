# PREVERVECT

High-fidelity Deepfake detection prototype with dual-stream spatial/frequency modeling.

## Project Structure

- `core/`: neural network architecture (SpecXNet)
- `utils/`: FFT tools and dataset preparation scripts
- `capture/`: realtime screen capture and inference
- `weights/`: pretrained and fine-tuned model weights
- `requirements.txt`: Python dependencies

## Environment Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Bootstrap the environment and download EfficientNet-B0 pretrained weights:

```bash
python setup_env.py
```

This script will:
- download/load `efficientnet_b0` via `timm`
- save its weights into `weights/efficientnet_b0_pretrained.pth`
- run a quick import/capture test for `mediapipe` and `mss`

## Realtime Detection

```bash
python capture/screen_detector.py
```

- Press `q` to quit safely.
- You can manually tune the `monitor` rectangle in `capture/screen_detector.py`
  (`top`, `left`, `width`, `height`) to align with your video playback window.

## FaceForensics++ Frame Preparation

```bash
python utils/prepare_data.py --data_root "path/to/FaceForensics-master" --output_root data/frames --every_n 10
```

Generated output folders:
- `data/frames/Real/...`
- `data/frames/Fake/...`
