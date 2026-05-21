"""Auto-estimate turf HSV bounds from broadcast frames."""

import cv2
import numpy as np

from src.config import (
    FIELD_HSV_AUTO_FRAMES,
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
)


def estimate_field_hsv(frames: list[np.ndarray]) -> tuple[int, int, int, int]:
    """Return (hue_low, hue_high, sat_low, val_low) from sample frames."""
    hues, sats, vals = [], [], []
    for frame in frames[:FIELD_HSV_AUTO_FRAMES]:
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.35) : int(h * 0.85), int(w * 0.1) : int(w * 0.9)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # green turf heuristic pre-filter
        mask = cv2.inRange(
            hsv,
            np.array([FIELD_HSV_HUE_LOW, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW]),
            np.array([FIELD_HSV_HUE_HIGH, 255, 255]),
        )
        pixels = hsv[mask > 0]
        if len(pixels) < 100:
            pixels = hsv.reshape(-1, 3)
        hues.extend(pixels[:, 0].tolist())
        sats.extend(pixels[:, 1].tolist())
        vals.extend(pixels[:, 2].tolist())

    if not hues:
        return FIELD_HSV_HUE_LOW, FIELD_HSV_HUE_HIGH, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW

    h_arr = np.array(hues)
    s_arr = np.array(sats)
    v_arr = np.array(vals)
    hue_low = int(max(0, np.percentile(h_arr, 5) - 5))
    hue_high = int(min(179, np.percentile(h_arr, 95) + 5))
    sat_low = int(max(20, np.percentile(s_arr, 10) - 10))
    val_low = int(max(20, np.percentile(v_arr, 10) - 10))
    return hue_low, hue_high, sat_low, val_low


def build_field_mask_from_bounds(
    frame: np.ndarray,
    hue_low: int,
    hue_high: int,
    sat_low: int,
    val_low: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([hue_low, sat_low, val_low])
    upper = np.array([hue_high, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
