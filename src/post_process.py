import cv2
import numpy as np

from src.config import (
    DUPLICATE_SUPPRESS_IOU,
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
    HUD_BOTTOM_PCT,
    HUD_FIELD_BOTTOM_MARGIN_PX,
    HUD_FIELD_TOP_MARGIN_PX,
    HUD_TOP_PCT,
    YARD_LINE_SAT_MAX,
    YARD_LINE_VAL_MIN,
)

_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))


def build_field_mask(
    frame: np.ndarray,
    *,
    hsv: np.ndarray | None = None,
    hue_low: int | None = None,
    hue_high: int | None = None,
    sat_low: int | None = None,
    val_low: int | None = None,
) -> np.ndarray:
    hl = FIELD_HSV_HUE_LOW if hue_low is None else hue_low
    hh = FIELD_HSV_HUE_HIGH if hue_high is None else hue_high
    sl = FIELD_HSV_SAT_LOW if sat_low is None else sat_low
    vl = FIELD_HSV_VAL_LOW if val_low is None else val_low
    if hsv is None:
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


def _hud_limits(frame_shape, field_mask: np.ndarray | None = None) -> tuple[float, float]:
    h = frame_shape[0]
    if field_mask is not None and field_mask.size:
        rows = np.where(field_mask.any(axis=1))[0]
        if len(rows):
            top_limit = max(0, int(rows[0]) - HUD_FIELD_TOP_MARGIN_PX)
            bottom_limit = min(h, int(rows[-1]) + HUD_FIELD_BOTTOM_MARGIN_PX)
            if bottom_limit > top_limit:
                return float(top_limit), float(bottom_limit)

    top_limit = h * HUD_TOP_PCT
    bottom_limit = h - (h * HUD_BOTTOM_PCT)
    return top_limit, bottom_limit


def filter_hud_detections(frame_shape, xyxy, confs, track_ids, *, field_mask: np.ndarray | None = None):
    if len(xyxy) == 0:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty, empty

    top_limit, bottom_limit = _hud_limits(frame_shape, field_mask=field_mask)
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    keep = (cy > top_limit) & (cy < bottom_limit)
    if not np.any(keep):
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty, empty
    idx = np.where(keep)[0]
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids, idx


def suppress_duplicate_tracks(
    detections: list[dict], iou_thresh: float = DUPLICATE_SUPPRESS_IOU
) -> list[dict]:
    """Remove lower-confidence tracks that heavily overlap a higher-confidence track."""
    if len(detections) <= 1:
        return detections
    sorted_dets = sorted(detections, key=lambda d: d["conf"], reverse=True)
    suppressed: set[int] = set()
    for i, det_i in enumerate(sorted_dets):
        if det_i["track_id"] in suppressed:
            continue
        x1i, y1i, x2i, y2i = det_i["bbox"]
        ai = max(0, x2i - x1i) * max(0, y2i - y1i)
        for det_j in sorted_dets[i + 1:]:
            if det_j["track_id"] in suppressed:
                continue
            x1j, y1j, x2j, y2j = det_j["bbox"]
            ix1, iy1 = max(x1i, x1j), max(y1i, y1j)
            ix2, iy2 = min(x2i, x2j), min(y2i, y2j)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            aj = max(0, x2j - x1j) * max(0, y2j - y1j)
            union = ai + aj - inter
            if union > 0 and inter / union > iou_thresh:
                suppressed.add(det_j["track_id"])
    return [d for d in sorted_dets if d["track_id"] not in suppressed]


def filter_by_field_area(
    frame: np.ndarray,
    xyxy,
    confs,
    track_ids,
    mask: np.ndarray,
    *,
    min_field_frac: float = 0.10,
):
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
        if foot_region.size > 0 and (np.count_nonzero(foot_region) / foot_region.size) > min_field_frac:
            keep.append(i)
    if not keep:
        empty = np.array([], dtype=int)
        return np.empty((0, 4)), np.array([]), empty
    idx = np.array(keep)
    tids = track_ids[idx] if track_ids is not None and len(track_ids) else None
    return xyxy[idx], confs[idx], tids
