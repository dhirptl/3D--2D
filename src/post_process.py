import numpy as np

from src.config import (
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
    HUD_BOTTOM_PCT,
    HUD_TOP_PCT,
)


def filter_hud_detections(frame_shape, xyxy, confs, track_ids):
    h, w = frame_shape[:2]
    top_limit = h * HUD_TOP_PCT
    bottom_limit = h - (h * HUD_BOTTOM_PCT)
    keep = []
    for i, box in enumerate(xyxy):
        cy = (box[1] + box[3]) / 2
        if top_limit < cy < bottom_limit:
            keep.append(i)
    if not keep:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty
    idx = np.array(keep)
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids


def build_field_mask(frame: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([FIELD_HSV_HUE_LOW, FIELD_HSV_SAT_LOW, FIELD_HSV_VAL_LOW])
    upper = np.array([FIELD_HSV_HUE_HIGH, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def filter_by_field_area(frame: np.ndarray, xyxy, confs, track_ids, mask: np.ndarray):
    keep = []
    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        box_h = y2 - y1
        bottom_y1 = int(y2 - (box_h * 0.15))
        foot_region = mask[
            max(0, bottom_y1) : min(frame.shape[0], y2),
            max(0, x1) : min(frame.shape[1], x2),
        ]
        if foot_region.size > 0 and (np.count_nonzero(foot_region) / foot_region.size) > 0.10:
            keep.append(i)
    if not keep:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty
    idx = np.array(keep)
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids
