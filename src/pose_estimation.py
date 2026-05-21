"""YOLO-Pose inference and matching to seg detections."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import POSE_IOU_MATCH, POSE_MODEL_PATH
from src.mask_utils import iou_matrix_xyxy

logger = logging.getLogger(__name__)

_pose_model: YOLO | None = None


def get_pose_model(weights: Path | None = None) -> YOLO:
    global _pose_model
    path = weights or POSE_MODEL_PATH
    if _pose_model is None:
        _pose_model = YOLO(str(path))
    return _pose_model


def run_pose_on_frame(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float = 0.25,
) -> list[dict]:
    """Return list of {bbox, keypoints (17,3)} in full-frame coords."""
    results = model.predict(frame, conf=conf, verbose=False)
    if not results or results[0].keypoints is None:
        return []
    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []
    boxes = r.boxes.xyxy.cpu().numpy()
    kpts = r.keypoints.data.cpu().numpy()  # (N, 17, 3)
    out = []
    for box, kp in zip(boxes, kpts):
        out.append({
            "bbox": tuple(map(int, box)),
            "keypoints": kp,
        })
    return out


def attach_keypoints_to_detections(
    detections: list[dict],
    pose_dets: list[dict],
    iou_thresh: float = POSE_IOU_MATCH,
) -> None:
    """Mutate detections in-place with matched keypoints (crop-relative)."""
    if not detections or not pose_dets:
        return

    det_boxes = np.array([d["bbox"] for d in detections], dtype=float)
    pose_boxes = np.array([p["bbox"] for p in pose_dets], dtype=float)
    iou = iou_matrix_xyxy(det_boxes, pose_boxes)

    for i, det in enumerate(detections):
        j = int(np.argmax(iou[i]))
        if iou[i, j] < iou_thresh:
            continue
        x1, y1, x2, y2 = map(int, det["bbox"])
        kp_full = pose_dets[j]["keypoints"].copy()
        kp_crop = kp_full.copy()
        kp_crop[:, 0] -= x1
        kp_crop[:, 1] -= y1
        det["keypoints"] = kp_crop
