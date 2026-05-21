"""Extract tracked detections with full-frame masks from YOLO-Seg + ByteTrack."""

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    PLAYER_CLASS_ID,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    TRACKER_CFG,
)
from src.post_process import filter_hud_detections

logger = logging.getLogger(__name__)


@dataclass
class DetectionStats:
    frames: int = 0
    total_dets: int = 0
    zero_det_frames: int = 0
    missing_mask_warnings: int = 0
    max_dets: int = 0
    per_frame_counts: list = field(default_factory=list)

    def record(self, n: int) -> None:
        self.frames += 1
        self.total_dets += n
        self.max_dets = max(self.max_dets, n)
        self.per_frame_counts.append(n)
        if n == 0:
            self.zero_det_frames += 1

    def summary(self) -> str:
        avg = self.total_dets / self.frames if self.frames else 0
        return (
            f"Detection summary: avg={avg:.1f}/frame max={self.max_dets} "
            f"zero_frames={self.zero_det_frames} mask_warnings={self.missing_mask_warnings}"
        )


def mask_to_full_frame(mask_tensor, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return (m > 0.5).astype(np.uint8) * 255


def mask_from_polygon(polygon_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon_xy is None or len(polygon_xy) < 3:
        return mask
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def get_instance_mask(result_masks, index: int, shape_hw: tuple[int, int]) -> np.ndarray | None:
    h, w = shape_hw
    if result_masks.data is not None and index < len(result_masks.data):
        return mask_to_full_frame(result_masks.data[index], (h, w))
    if hasattr(result_masks, "xy") and result_masks.xy is not None and index < len(result_masks.xy):
        poly = result_masks.xy[index]
        if poly is not None and len(poly) >= 3:
            return mask_from_polygon(np.array(poly), (h, w))
    return None


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_indices(filtered_xyxy: np.ndarray, orig_xyxy: np.ndarray) -> list[int]:
    """Map each filtered box to best original index by IoU."""
    indices = []
    for box in filtered_xyxy:
        ious = [box_iou_xyxy(box, ob) for ob in orig_xyxy]
        indices.append(int(np.argmax(ious)))
    return indices


def extract_tracked_detections(
    model: YOLO,
    frame: np.ndarray,
    *,
    frame_idx: int = 0,
    apply_hud_filter: bool = True,
    stats: DetectionStats | None = None,
) -> list[dict]:
    tracker = str(TRACKER_CFG) if TRACKER_CFG.exists() else "bytetrack.yaml"
    results = model.track(
        frame,
        classes=[PLAYER_CLASS_ID],
        conf=PLAYER_PREDICT_CONF,
        iou=PLAYER_PREDICT_IOU,
        imgsz=PLAYER_IMGSZ,
        max_det=PLAYER_PREDICT_MAX_DET,
        tracker=tracker,
        persist=True,
        retina_masks=False,
        verbose=False,
    )
    result = results[0]

    if result.boxes is None or result.boxes.id is None:
        if stats:
            stats.record(0)
        return []

    if result.masks is None:
        if stats:
            stats.missing_mask_warnings += 1
            stats.record(0)
        logger.warning("frame %d: boxes present but masks is None", frame_idx)
        return []

    h, w = frame.shape[:2]
    orig_xyxy = result.boxes.xyxy.cpu().numpy()
    xyxy = orig_xyxy.copy()
    confs = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    result_masks = result.masks

    if apply_hud_filter:
        xyxy, confs, track_ids = filter_hud_detections(
            frame.shape, xyxy, confs, track_ids
        )
        if len(xyxy) == 0:
            if stats:
                stats.record(0)
            return []

    keep_idx = match_indices(xyxy, orig_xyxy)

    detections = []
    for i, k in enumerate(keep_idx):
        mask = get_instance_mask(result_masks, k, (h, w))
        if mask is None or np.count_nonzero(mask) == 0:
            if stats:
                stats.missing_mask_warnings += 1
            continue
        box = tuple(map(int, xyxy[i]))
        detections.append({
            "track_id": int(track_ids[i]),
            "bbox": box,
            "conf": float(confs[i]),
            "mask": mask,
        })

    if stats:
        stats.record(len(detections))
    return detections
