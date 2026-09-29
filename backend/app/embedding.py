"""Deterministic appearance descriptor used by offline and online retrieval."""

from __future__ import annotations

import cv2
import numpy as np

EMBEDDING_DIM = 386
INPUT_SIZE = (128, 128)
PLATE_MASK_TOP_PERCENT = 58
PLATE_MASK_LEFT_PERCENT = 18
PLATE_MASK_RIGHT_PERCENT = 82


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("embedding input must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(values))
    if norm < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return (values / norm).astype(np.float32, copy=False)


def cosine_confidence(score: float) -> float:
    """Map cosine similarity to [0, 1]; this is not probability calibration."""
    return min(1.0, max(0.0, (score + 1.0) / 2.0))


def vehicle_embedding(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract a 386-value L2-normalized feature vector from a BGR vehicle crop."""
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or crop_bgr.size == 0:
        raise ValueError("crop must be a non-empty three-channel BGR image")

    image = cv2.resize(crop_bgr, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    parts: list[np.ndarray] = []
    for channel, bins, upper in ((0, 32, 180), (1, 32, 256), (2, 32, 256)):
        histogram = cv2.calcHist([hsv], [channel], None, [bins], [0, upper]).ravel()
        parts.append(histogram / max(float(histogram.sum()), 1.0))

    for grid_y, grid_x in ((4, 4), (8, 4)):
        for cell in np.array_split(image, grid_y, axis=0):
            for patch in np.array_split(cell, grid_x, axis=1):
                parts.extend(
                    (
                        patch.mean(axis=(0, 1)),
                        patch.std(axis=(0, 1)),
                    )
                )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
    parts.append(np.asarray([edges.mean(), edges.std()], dtype=np.float32))
    embedding = l2_normalize(
        np.concatenate([np.asarray(part, dtype=np.float32).ravel() for part in parts])
    )
    if embedding.shape != (EMBEDDING_DIM,):
        raise RuntimeError(f"expected {EMBEDDING_DIM} features; got {embedding.size}")
    return embedding


def mask_sensitive_regions(crop_bgr: np.ndarray) -> np.ndarray:
    """Mask a fixed lower-central area where front/rear plates are commonly located.

    This is a geometric heuristic, not plate or face detection. Source images
    must still meet the dataset's anonymization requirements.
    """
    crop = crop_bgr.copy()
    mask_top = crop.shape[0] * PLATE_MASK_TOP_PERCENT // 100
    mask_left = crop.shape[1] * PLATE_MASK_LEFT_PERCENT // 100
    mask_right = crop.shape[1] * PLATE_MASK_RIGHT_PERCENT // 100
    crop[mask_top:, mask_left:mask_right] = 0
    return crop
