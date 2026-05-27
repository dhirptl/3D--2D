"""Auto-estimate turf HSV bounds from broadcast frames."""

import cv2
import numpy as np

from src.config import (
    FIELD_HSV_AUTO_FRAMES,
    FIELD_HSV_CALIB_PX_PER_FRAME,
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_MAX_PIXELS,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
    FIELD_ROI_X0_FRAC,
    FIELD_ROI_X1_FRAC,
    FIELD_ROI_Y0_FRAC,
    FIELD_ROI_Y1_FRAC,
)
from src.post_process import build_field_mask


def _subsample_pixels(pixels: np.ndarray, max_n: int) -> np.ndarray:
    if len(pixels) <= max_n:
        return pixels
    idx = np.random.choice(len(pixels), max_n, replace=False)
    return pixels[idx]


def _hsv_bounds_from_pixels(all_px: np.ndarray) -> tuple[int, int, int, int]:
    hue_low  = int(max(0,   np.percentile(all_px[:, 0],  5) - 5))
    hue_high = int(min(179, np.percentile(all_px[:, 0], 95) + 5))
    sat_low  = int(max(20,  np.percentile(all_px[:, 1], 10) - 10))
    val_low  = int(max(20,  np.percentile(all_px[:, 2], 10) - 10))
    return hue_low, hue_high, sat_low, val_low


def estimate_hsv_from_pixel_arrays(
    pixel_arrays: list[np.ndarray],
) -> tuple[int, int, int, int]:
    """Return (hue_low, hue_high, sat_low, val_low) from pre-sampled HSV pixel arrays."""
    if not pixel_arrays:
        return FIELD_HSV_HUE_LOW, FIELD_HSV_HUE_HIGH, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW
    all_px = np.concatenate(pixel_arrays, axis=0)
    return _hsv_bounds_from_pixels(all_px)


def estimate_field_hsv(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Return (hue_low, hue_high, sat_low, val_low) from sample frames."""
    chunks: list[np.ndarray] = []
    per_frame_cap = max(100, FIELD_HSV_MAX_PIXELS // FIELD_HSV_AUTO_FRAMES)

    for frame in frames[:FIELD_HSV_AUTO_FRAMES]:
        h, w = frame.shape[:2]
        roi = frame[
            int(h * FIELD_ROI_Y0_FRAC) : int(h * FIELD_ROI_Y1_FRAC),
            int(w * FIELD_ROI_X0_FRAC) : int(w * FIELD_ROI_X1_FRAC),
        ]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array([FIELD_HSV_HUE_LOW, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW]),
            np.array([FIELD_HSV_HUE_HIGH, 255, 255]),
        )
        pixels = hsv[mask > 0]
        if len(pixels) < 100:
            pixels = hsv.reshape(-1, 3)
        chunks.append(_subsample_pixels(pixels, per_frame_cap))

    if not chunks:
        return FIELD_HSV_HUE_LOW, FIELD_HSV_HUE_HIGH, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW
    return _hsv_bounds_from_pixels(np.concatenate(chunks, axis=0))


def build_field_mask_from_bounds(
    frame: np.ndarray,
    hue_low: int,
    hue_high: int,
    sat_low: int,
    val_low: int,
    *,
    hsv: np.ndarray | None = None,
) -> np.ndarray:
    return build_field_mask(
        frame, hsv=hsv, hue_low=hue_low, hue_high=hue_high, sat_low=sat_low, val_low=val_low
    )
