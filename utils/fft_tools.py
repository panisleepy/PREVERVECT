"""FFT utilities for frequency-domain feature generation."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def fft_log_magnitude_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR frame into a normalized 3-channel log-magnitude spectrum.

    Returns:
        np.ndarray with shape [H, W, 3], float32 in [0, 1].
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    log_mag = np.log1p(magnitude)
    log_mag = cv2.normalize(log_mag, None, 0.0, 1.0, cv2.NORM_MINMAX)
    spectrum = np.repeat(log_mag[..., None], 3, axis=2).astype(np.float32)
    return spectrum


def bgr_to_tensor(frame_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Resize + normalize BGR image to PyTorch tensor [1, 3, size, size].
    """
    resized = cv2.resize(frame_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0)
    return tensor


def fft_to_tensor(frame_bgr: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Convert BGR frame to FFT spectrum tensor [1, 3, size, size].
    """
    spectrum = fft_log_magnitude_bgr(frame_bgr)
    resized = cv2.resize(spectrum, (size, size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(np.transpose(resized, (2, 0, 1))).unsqueeze(0)
    return tensor


def power_spectrum_shifted_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to centered 2D power spectrum image.

    Steps:
      1) grayscale
      2) 2D FFT
      3) shift DC component to center
      4) power spectrum = |F(u,v)|^2
      5) log compress and normalize to [0, 1]

    Returns:
        np.ndarray [H, W, 3], float32 in [0, 1].
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    power = np.abs(fshift) ** 2
    power_log = np.log1p(power)
    power_norm = cv2.normalize(power_log, None, 0.0, 1.0, cv2.NORM_MINMAX)
    spectrum = np.repeat(power_norm[..., None], 3, axis=2).astype(np.float32)
    return spectrum
