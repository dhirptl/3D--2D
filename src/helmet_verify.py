"""Helmet detection helpers for player verification."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from src.config import (
    HEAD_ROI_FRAC,
    HELMET_CONF,
    HELMET_GATE_GRACE,
    HELMET_GATE_LOCKED_REQUIRED,
    HELMET_GATE_REQUIRED,
    HELMET_GATE_SOFT_LOCK_STREAK,
    HELMET_GATE_WINDOW,
    HELMET_HEAD_IOU,
)


def _clip_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    bounds: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    bx1, by1, bx2, by2 = bounds
    x1 = max(bx1, min(int(round(x1)), bx2))
    y1 = max(by1, min(int(round(y1)), by2))
    x2 = max(bx1, min(int(round(x2)), bx2))
    y2 = max(by1, min(int(round(y2)), by2))
    if x2 <= x1:
        x2 = min(bx2, x1 + 1)
    if y2 <= y1:
        y2 = min(by2, y1 + 1)
    return x1, y1, x2, y2


def head_roi(
    bbox: tuple[int, int, int, int],
    keypoints: np.ndarray | None = None,
    *,
    top_frac: float = HEAD_ROI_FRAC,
    min_kpt_conf: float = 0.3,
) -> tuple[int, int, int, int]:
    """Estimate a head ROI in full-frame coordinates."""
    x1, y1, x2, y2 = map(int, bbox)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    fallback = (x1, y1, x2, y1 + max(1, int(box_h * top_frac)))

    if keypoints is None or len(keypoints) < 3:
        return fallback

    pts = []
    for idx in (0, 1, 2):  # nose, left eye, right eye
        kx, ky, conf = keypoints[idx]
        if conf >= min_kpt_conf:
            pts.append((x1 + float(kx), y1 + float(ky)))
    if not pts:
        return fallback

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)

    spread_x = (max(xs) - min(xs)) if len(xs) > 1 else 0.0
    half_w = max(box_w * 0.18, spread_x * 1.2, 6.0)
    half_h = max(box_h * 0.10, 6.0)

    return _clip_box(
        cx - half_w,
        cy - (half_h * 1.2),
        cx + half_w,
        cy + (half_h * 1.4),
        (x1, y1, x2, y2),
    )


def detect_helmets(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float = HELMET_CONF,
) -> list[dict]:
    """Run the helmet model once on the full frame."""
    results = model.predict(frame, conf=conf, verbose=False)
    if not results:
        return []
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    helmets = []
    for box, score in zip(boxes, confs):
        helmets.append({
            "bbox": tuple(map(int, box)),
            "conf": float(score),
        })
    return helmets


def detect_helmets_roi(
    model: YOLO,
    frame: np.ndarray,
    detections: list[dict],
    *,
    conf: float = HELMET_CONF,
    pad_frac: float = 0.5,
) -> list[dict]:
    """Run the helmet model on padded head crops and return full-frame boxes."""
    if not detections:
        return []

    fh, fw = frame.shape[:2]
    crops: list[np.ndarray] = []
    offsets: list[tuple[int, int, int, int]] = []

    for det in detections:
        x1, y1, x2, y2 = head_roi(det["bbox"], det.get("keypoints"))
        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)
        pad_x = int(roi_w * pad_frac)
        pad_y = int(roi_h * pad_frac)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(fw, x2 + pad_x)
        cy2 = min(fh, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        crops.append(crop)
        offsets.append((cx1, cy1, cx2, cy2))

    if not crops:
        return []

    results = model.predict(crops, conf=conf, verbose=False)
    helmets: list[dict] = []
    for result, (ox1, oy1, ox2, oy2) in zip(results, offsets):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        for box, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            bx1, by1, bx2, by2 = _clip_box(
                float(box[0]) + ox1,
                float(box[1]) + oy1,
                float(box[2]) + ox1,
                float(box[3]) + oy1,
                (0, 0, fw, fh),
            )
            if bx2 <= bx1 or by2 <= by1:
                continue
            helmets.append({
                "bbox": (bx1, by1, bx2, by2),
                "conf": float(score),
            })
    return helmets


def _box_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0, ix2 - ix1)
    inter_h = max(0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_inside(
    roi: tuple[int, int, int, int],
    box: tuple[int, int, int, int],
) -> bool:
    rx1, ry1, rx2, ry2 = roi
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def helmet_confirms_player(
    roi: tuple[int, int, int, int],
    helmets: list[dict],
    *,
    conf_min: float = HELMET_CONF,
    iou_min: float = HELMET_HEAD_IOU,
) -> bool:
    """Confirm a player when a helmet center falls in the head ROI."""
    for helmet in helmets:
        if helmet["conf"] < conf_min:
            continue
        box = helmet["bbox"]
        if _center_inside(roi, box):
            return True
        if iou_min > 0 and _box_iou(roi, box) >= iou_min:
            return True
    return False


@dataclass
class HelmetTrackGate:
    window: int = HELMET_GATE_WINDOW
    required: int = HELMET_GATE_REQUIRED
    grace: int = HELMET_GATE_GRACE
    soft_lock_streak: int = HELMET_GATE_SOFT_LOCK_STREAK
    locked_required: int = HELMET_GATE_LOCKED_REQUIRED
    history: dict[int, deque[bool]] = field(
        default_factory=lambda: defaultdict(lambda: deque(maxlen=HELMET_GATE_WINDOW))
    )
    seen: dict[int, int] = field(default_factory=dict)
    confirmed_streak: dict[int, int] = field(default_factory=dict)
    soft_locked: dict[int, bool] = field(default_factory=dict)
    last_decision: dict[int, bool] = field(default_factory=dict)

    def observe(self, track_id: int, confirmed: bool) -> bool:
        hist = self.history[track_id]
        if hist.maxlen != self.window:
            hist = deque(hist, maxlen=self.window)
            self.history[track_id] = hist
        hist.append(bool(confirmed))
        seen = self.seen.get(track_id, 0) + 1
        self.seen[track_id] = seen

        streak = self.confirmed_streak.get(track_id, 0)
        if confirmed:
            streak += 1
        else:
            streak = 0
        self.confirmed_streak[track_id] = streak
        if streak >= self.soft_lock_streak:
            self.soft_locked[track_id] = True

        confirmed_count = sum(hist)
        if self.soft_locked.get(track_id) and confirmed_count == 0:
            self.soft_locked.pop(track_id, None)

        required = self.locked_required if self.soft_locked.get(track_id, False) else self.required
        decision = seen <= self.grace or confirmed_count >= required
        self.last_decision[track_id] = decision
        return decision

    def current(self, track_id: int, confirmed: bool) -> bool:
        if track_id in self.last_decision:
            return self.last_decision[track_id]
        if self.grace > 0:
            return True
        return confirmed

    def prune(self, active_track_ids: set[int]) -> None:
        stale = set(self.last_decision) - active_track_ids
        for track_id in stale:
            self.history.pop(track_id, None)
            self.seen.pop(track_id, None)
            self.confirmed_streak.pop(track_id, None)
            self.soft_locked.pop(track_id, None)
            self.last_decision.pop(track_id, None)


def filter_detections_by_helmet(
    detections: list[dict],
    helmets: list[dict],
    *,
    gate: HelmetTrackGate | None = None,
    conf_min: float = HELMET_CONF,
    iou_min: float = HELMET_HEAD_IOU,
    update_gate: bool = True,
) -> list[dict]:
    """Keep only detections with a matching helmet near the head region."""
    filtered: list[dict] = []
    active_ids: set[int] = set()

    for det in detections:
        track_id = int(det["track_id"])
        active_ids.add(track_id)
        roi = head_roi(det["bbox"], det.get("keypoints"))
        confirmed = helmet_confirms_player(roi, helmets, conf_min=conf_min, iou_min=iou_min)

        if gate is None:
            allowed = confirmed
        elif update_gate:
            allowed = gate.observe(track_id, confirmed)
        else:
            allowed = gate.current(track_id, confirmed)

        updated = {
            **det,
            "head_roi": roi,
            "helmet_confirmed": confirmed,
            "helmet_allowed": allowed,
        }
        if allowed:
            filtered.append(updated)

    if gate is not None:
        gate.prune(active_ids)
    return filtered
