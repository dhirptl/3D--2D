"""Tiled inference helpers: merge, WBF (DESIGN §4.2). SAHI optional at runtime."""

from __future__ import annotations

import numpy as np


def tile_slices(
    width: int,
    height: int,
    tile_size: int = 640,
    overlap: float = 0.2,
) -> list[tuple[int, int, int, int]]:
    """Return (x1, y1, x2, y2) slices covering the frame."""
    step = max(1, int(tile_size * (1.0 - overlap)))
    boxes: list[tuple[int, int, int, int]] = []
    for y in range(0, height, step):
        for x in range(0, width, step):
            x2 = min(width, x + tile_size)
            y2 = min(height, y + tile_size)
            x1 = max(0, x2 - tile_size)
            y1 = max(0, y2 - tile_size)
            boxes.append((x1, y1, x2, y2))
    return boxes


def map_boxes_to_full(
    boxes: np.ndarray,
    slice_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, _, _ = slice_xyxy
    out = boxes.copy()
    out[:, [0, 2]] += x1
    out[:, [1, 3]] += y1
    return out


def nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.5,
) -> np.ndarray:
    """Return indices to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _iou_vec(boxes[i], boxes[rest])
        order = rest[ious <= iou_thresh]
    return np.array(keep, dtype=int)


def _iou_vec(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    a = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    union = a + areas - inter
    return inter / np.maximum(union, 1e-6)


def weighted_box_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    *,
    iou_thresh: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse multiple box sets by averaging clusters (WBF-lite)."""
    if not boxes_list:
        return np.zeros((0, 4)), np.zeros(0)
    boxes = np.vstack(boxes_list)
    scores = np.concatenate(scores_list)
    keep = nms_xyxy(boxes, scores, iou_thresh=iou_thresh)
    if len(keep) == 0:
        return np.zeros((0, 4)), np.zeros(0)
    fused_boxes = []
    fused_scores = []
    used = set()
    for i in keep:
        if i in used:
            continue
        cluster = [i]
        used.add(i)
        for j in keep:
            if j in used:
                continue
            if _iou_vec(boxes[i], boxes[j : j + 1])[0] >= iou_thresh:
                cluster.append(j)
                used.add(j)
        w = scores[cluster]
        fused_boxes.append(np.average(boxes[cluster], axis=0, weights=w))
        fused_scores.append(float(w.mean()))
    return np.array(fused_boxes), np.array(fused_scores)


def merge_tile_detections(
    all_boxes: np.ndarray,
    all_scores: np.ndarray,
    *,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """NMS merge after tiled inference."""
    keep = nms_xyxy(all_boxes, all_scores, iou_thresh=iou_thresh)
    return all_boxes[keep], all_scores[keep]
