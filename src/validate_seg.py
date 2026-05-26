"""Gate 2: validate a segmentation model on the active Stage 1 dataset."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from src.config import (
    GATE2_BOX_RECALL,
    GATE2_MASK_MAP50,
    GATE2_MASK_MAP75,
    GATE2_MASK_RECALL,
    PLAYER_IMGSZ,
    SEG_DATASET_YAML,
    SEG_MODEL_PATH,
)

_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
_SMALL_AREA = 32 * 32
_MEDIUM_AREA = 96 * 96


def _scalar(val) -> float:
    arr = np.asarray(val)
    return float(arr.mean() if arr.size > 1 else arr.item())


def _polygon_line_to_box(line: str, w: int, h: int) -> tuple[float, float, float, float] | None:
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        return None
    pts = np.asarray(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
    xs = pts[:, 0] * w
    ys = pts[:, 1] * h
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _size_bucket(area: float) -> str:
    if area < _SMALL_AREA:
        return "small"
    if area < _MEDIUM_AREA:
        return "medium"
    return "large"


def _size_bucket_recall(model: YOLO, data_yaml: Path, *, imgsz: int = PLAYER_IMGSZ) -> dict[str, dict[str, float]]:
    cfg = yaml.safe_load(data_yaml.read_text())
    root = Path(cfg["path"]).resolve()
    val_images = root / cfg["val"]
    val_labels = val_images.parent.parent / "labels"
    stats = {name: {"gt": 0, "matched": 0} for name in ("small", "medium", "large")}

    for img_path in sorted(val_images.iterdir()):
        if img_path.suffix.lower() not in _IMAGE_EXTS:
            continue
        label_path = val_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        gt_boxes = []
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            box = _polygon_line_to_box(line, w, h)
            if box is not None:
                gt_boxes.append(box)
        if not gt_boxes:
            continue

        result = model.predict(
            image,
            imgsz=imgsz,
            conf=0.001,
            iou=0.7,
            max_det=300,
            verbose=False,
        )[0]
        pred_boxes = []
        if result.boxes is not None and len(result.boxes):
            pred_boxes = [tuple(map(float, box)) for box in result.boxes.xyxy.cpu().numpy()]

        used_preds: set[int] = set()
        for gt_box in gt_boxes:
            area = max(0.0, gt_box[2] - gt_box[0]) * max(0.0, gt_box[3] - gt_box[1])
            bucket = _size_bucket(area)
            stats[bucket]["gt"] += 1
            best_idx = -1
            best_iou = 0.0
            for idx, pred_box in enumerate(pred_boxes):
                if idx in used_preds:
                    continue
                iou = _box_iou(gt_box, pred_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= 0.5 and best_idx >= 0:
                used_preds.add(best_idx)
                stats[bucket]["matched"] += 1

    out: dict[str, dict[str, float]] = {}
    for bucket, bucket_stats in stats.items():
        gt = bucket_stats["gt"]
        matched = bucket_stats["matched"]
        out[bucket] = {
            "gt": gt,
            "matched": matched,
            "recall": round(matched / gt, 3) if gt else 0.0,
        }
    return out


def validate(
    weights: Path | None = None,
    data_yaml: Path | None = None,
    *,
    include_size_diagnostics: bool = True,
) -> bool:
    w = weights or SEG_MODEL_PATH
    data = data_yaml or SEG_DATASET_YAML
    if not w.exists():
        raise FileNotFoundError(f"Weights not found: {w}")
    if not data.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data}")

    model = YOLO(str(w))
    metrics = model.val(data=str(data), verbose=False)

    box_map50 = _scalar(metrics.box.map50)
    box_recall = _scalar(metrics.box.r)
    mask_map50 = _scalar(metrics.seg.map50)
    mask_map75 = _scalar(getattr(metrics.seg, "map75", 0.0))
    mask_recall = _scalar(metrics.seg.r)

    print("--- Gate 2 Validation ---")
    box_mp = _scalar(metrics.box.mp)
    mask_mp = _scalar(metrics.seg.mp)
    print(f"  Weights: {w}")
    print(f"  Dataset: {data}")
    print(f"  Box  P/R/mAP50:     {box_mp:.3f} / {box_recall:.3f} / {box_map50:.3f}")
    print(f"  Mask P/R/mAP50/75: {mask_mp:.3f} / {mask_recall:.3f} / {mask_map50:.3f} / {mask_map75:.3f}")

    print(f"  Box recall target (>= {GATE2_BOX_RECALL:.2f}): {'PASS' if box_recall >= GATE2_BOX_RECALL else 'FAIL'}")
    print(f"  Mask mAP50 target (>= {GATE2_MASK_MAP50:.2f}): {'PASS' if mask_map50 >= GATE2_MASK_MAP50 else 'FAIL'}")
    print(f"  Mask mAP75 target (>= {GATE2_MASK_MAP75:.2f}): {'PASS' if mask_map75 >= GATE2_MASK_MAP75 else 'FAIL'}")
    print(f"  Mask recall target (>= {GATE2_MASK_RECALL:.2f}): {'PASS' if mask_recall >= GATE2_MASK_RECALL else 'FAIL'}")

    if include_size_diagnostics:
        size_stats = _size_bucket_recall(model, data, imgsz=PLAYER_IMGSZ)
        print("  Size diagnostics (box recall@0.5 by GT area):")
        for bucket in ("small", "medium", "large"):
            bucket_stats = size_stats[bucket]
            print(
                f"    {bucket:6s}: recall={bucket_stats['recall']:.3f} "
                f"(matched={bucket_stats['matched']}, gt={bucket_stats['gt']})"
            )

    passed = (
        box_recall >= GATE2_BOX_RECALL
        and mask_map50 >= GATE2_MASK_MAP50
        and mask_map75 >= GATE2_MASK_MAP75
        and mask_recall >= GATE2_MASK_RECALL
    )
    print(f"  Gate 2 overall: {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--skip-size-diagnostics", action="store_true")
    args = parser.parse_args()
    ok = validate(
        Path(args.weights) if args.weights else None,
        Path(args.data) if args.data else None,
        include_size_diagnostics=not args.skip_size_diagnostics,
    )
    sys.exit(0 if ok else 1)
