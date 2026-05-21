import cv2
import numpy as np

from src.config import (
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
    HUD_BOTTOM_PCT,
    HUD_TOP_PCT,
    YARD_LINE_SAT_MAX,
    YARD_LINE_VAL_MIN,
)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))


def build_field_mask(
    frame: np.ndarray,
    *,
    hue_low: int | None = None,
    hue_high: int | None = None,
    sat_low: int | None = None,
    val_low: int | None = None,
) -> np.ndarray:
    hl = FIELD_HSV_HUE_LOW if hue_low is None else hue_low
    hh = FIELD_HSV_HUE_HIGH if hue_high is None else hue_high
    sl = FIELD_HSV_SAT_LOW if sat_low is None else sat_low
    vl = FIELD_HSV_VAL_LOW if val_low is None else val_low
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([hl, sl, vl])
    upper = np.array([hh, 255, 255])
    green = cv2.inRange(hsv, lower, upper)
    # White/yellow yard lines: low saturation, high value
    line_lower = np.array([0, 0, YARD_LINE_VAL_MIN])
    line_upper = np.array([179, YARD_LINE_SAT_MAX, 255])
    lines = cv2.inRange(hsv, line_lower, line_upper)
    mask = cv2.bitwise_or(green, lines)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)


def filter_hud_detections(frame_shape, xyxy, confs, track_ids):
    if len(xyxy) == 0:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty

    h = frame_shape[0]
    top_limit = h * HUD_TOP_PCT
    bottom_limit = h - (h * HUD_BOTTOM_PCT)
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    keep = (cy > top_limit) & (cy < bottom_limit)
    if not np.any(keep):
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty
    idx = np.where(keep)[0]
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids


def filter_by_field_area(frame: np.ndarray, xyxy, confs, track_ids, mask: np.ndarray):
    if len(xyxy) == 0:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty

    fh, fw = frame.shape[:2]
    keep = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        box_h = y2 - y1
        bottom_y1 = int(y2 - (box_h * 0.15))
        foot_region = mask[
            max(0, bottom_y1) : min(fh, y2),
            max(0, x1) : min(fw, x2),
        ]
        if foot_region.size > 0 and (np.count_nonzero(foot_region) / foot_region.size) > 0.10:
            keep.append(i)
    if not keep:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty
    idx = np.array(keep)
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids
