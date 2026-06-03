"""Evaluate box detection metrics on a bbox-only YOLO dataset.

Replaces validate_seg.py for datasets that have bounding-box labels only
(no polygon masks). Reports box P/R/mAP50/mAP50-95 + size-bucketed recall.

Usage:
  python scripts/eval_detection.py \
      --weights football_tracker_seg/run_v3-4_hardneg/weights/best.pt \
      --data data/football_eval/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.config import PLAYER_IMGSZ, SEG_MODEL_PATH

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SMALL_AREA = 32 * 32
_MEDIUM_AREA = 96 * 96


def _parse_bbox_line(line: str, w: int, h: int):
    """YOLO bbox format: class cx cy bw bh (all normalised)."""
    parts = line.split()
    if len(parts) != 5:
        return None
    cx, cy, bw, bh = map(float, parts[1:])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return x1, y1, x2, y2


def _box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _size_bucket(area: float) -> str:
    return "small" if area < _SMALL_AREA else ("medium" if area < _MEDIUM_AREA else "large")


def evaluate(weights: Path, data_yaml: Path, *, imgsz: int = PLAYER_IMGSZ) -> None:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"]).resolve()

    # Use 'test' split if present, otherwise 'val'
    split_key = "test" if "test" in cfg else "val"
    val_images = root / cfg[split_key]
    val_labels = val_images.parent / "labels"

    model = YOLO(str(weights))

    # Collect per-image predictions and GT — model.val() crashes on bbox-only
    # datasets when weights are from a seg model, so we evaluate manually.
    all_preds: list[tuple[float, bool]] = []  # (confidence, is_tp)
    total_gt = 0

    images = sorted(p for p in val_images.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    print(f"Evaluating {len(images)} images ({split_key} split) ...")
    for img_path in images:
        lbl = val_labels / f"{img_path.stem}.txt"
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = [
            b for line in (lbl.read_text().splitlines() if lbl.exists() else [])
            if line.strip()
            for b in [_parse_bbox_line(line, w, h)]
            if b is not None
        ]
        total_gt += len(gt_boxes)
        result = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.6,
                               max_det=300, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue
        preds_xyxy = [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]
        preds_conf = result.boxes.conf.cpu().numpy().tolist()
        matched_gt: set[int] = set()
        for conf, pred in sorted(zip(preds_conf, preds_xyxy), reverse=True):
            best_iou, best_gi = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = _box_iou(pred, gt)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            is_tp = best_iou >= 0.5 and best_gi >= 0
            if is_tp:
                matched_gt.add(best_gi)
            all_preds.append((conf, is_tp))

    # Compute precision/recall curve → AP@50 (already at IoU=0.5)
    all_preds.sort(key=lambda x: x[0], reverse=True)
    tp_cumsum = np.cumsum([int(tp) for _, tp in all_preds])
    fp_cumsum = np.cumsum([int(not tp) for _, tp in all_preds])
    recalls    = tp_cumsum / max(total_gt, 1)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)

    # AP = area under PR curve (trapezoidal)
    ap50 = float(np.trapezoid(precisions, recalls)) if len(recalls) > 1 else 0.0
    final_recall    = float(recalls[-1])    if len(recalls) else 0.0
    final_precision = float(precisions[-1]) if len(precisions) else 0.0

    print()
    print(f"Weights : {weights}")
    print(f"Dataset : {data_yaml}  [{split_key}, {len(images)} images, {total_gt} GT boxes]")
    print(f"  Box Precision : {final_precision:.3f}")
    print(f"  Box Recall    : {final_recall:.3f}")
    print(f"  mAP@50        : {ap50:.3f}")

    # Size-bucketed recall (own loop — handles plain bbox labels)
    stats = {b: {"gt": 0, "matched": 0} for b in ("small", "medium", "large")}
    for img_path in sorted(val_images.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        lbl = val_labels / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt_boxes = [
            b for line in lbl.read_text().splitlines()
            if line.strip()
            for b in [_parse_bbox_line(line, w, h)]
            if b is not None
        ]
        if not gt_boxes:
            continue
        result = model.predict(img, imgsz=imgsz, conf=0.001, iou=0.6,
                               max_det=300, verbose=False)[0]
        preds = (
            [tuple(map(float, b)) for b in result.boxes.xyxy.cpu().numpy()]
            if result.boxes is not None and len(result.boxes) else []
        )
        used: set[int] = set()
        for gt in gt_boxes:
            area = max(0.0, gt[2]-gt[0]) * max(0.0, gt[3]-gt[1])
            bucket = _size_bucket(area)
            stats[bucket]["gt"] += 1
            best_i, best_iou = -1, 0.0
            for i, p in enumerate(preds):
                if i in used:
                    continue
                iou = _box_iou(gt, p)
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_iou >= 0.5 and best_i >= 0:
                used.add(best_i)
                stats[bucket]["matched"] += 1

    print("  Size recall@IoU0.5:")
    for b in ("small", "medium", "large"):
        s = stats[b]
        r = s["matched"] / s["gt"] if s["gt"] else 0.0
        print(f"    {b:6s}: {r:.3f}  (matched {s['matched']}/{s['gt']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(SEG_MODEL_PATH))
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    evaluate(Path(args.weights), Path(args.data))


if __name__ == "__main__":
    main()
