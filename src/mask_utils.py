"""Mask cropping, area, and vectorized bbox IoU matching."""

import cv2
import numpy as np


def mask_area(mask: np.ndarray) -> int:
    """Pixel count for uint8 crop mask (0/255)."""
    return int(np.count_nonzero(mask))


def erode_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode binary mask inward to drop boundary/turf pixels."""
    if px <= 0 or not np.any(mask):
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
    return cv2.erode(mask, k)


def crop_mask_to_bbox(full_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = map(int, bbox)
    return full_mask[y1:y2, x1:x2].copy()


def iou_matrix_xyxy(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix (len_a, len_b)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_indices(filtered_xyxy: np.ndarray, orig_xyxy: np.ndarray) -> list[int]:
    """Greedy unique assignment: each orig index used at most once."""
    n_f, n_o = len(filtered_xyxy), len(orig_xyxy)
    if n_f == 0:
        return []
    if n_o == 0:
        return [0] * n_f

    iou = iou_matrix_xyxy(filtered_xyxy, orig_xyxy)
    used_orig: set[int] = set()
    indices: list[int] = []
    for i in range(n_f):
        best = int(np.argmax(iou[i]))
        if best not in used_orig:
            indices.append(best)
            used_orig.add(best)
            continue
        # Best already taken — fall back to sorted scan
        order = np.argsort(-iou[i])
        chosen = int(order[0])
        for j in order:
            j = int(j)
            if j not in used_orig:
                chosen = j
                used_orig.add(j)
                break
        indices.append(chosen)
    return indices


def mask_to_full_frame(mask_tensor, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    m = mask_tensor.cpu().numpy()
    if m.ndim == 3:
        m = m[0]
    if m.shape != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    # Ultralytics may emit binary masks as 0/1 or 0/255 depending on backend/version.
    thresh = 0.5 if float(np.max(m)) <= 1.0 else 127.0
    return (m > thresh).astype(np.uint8) * 255


def mask_from_polygon(polygon_xy: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon_xy is None or len(polygon_xy) < 3:
        return mask
    pts = np.array(polygon_xy, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def get_instance_mask_full(
    result_masks, index: int, shape_hw: tuple[int, int]
) -> np.ndarray | None:
    h, w = shape_hw
    if result_masks.data is not None and index < len(result_masks.data):
        return mask_to_full_frame(result_masks.data[index], (h, w))
    if hasattr(result_masks, "xy") and result_masks.xy is not None and index < len(result_masks.xy):
        poly = result_masks.xy[index]
        if poly is not None and len(poly) >= 3:
            return mask_from_polygon(np.array(poly), (h, w))
    return None
