"""Generate YOLO segmentation labels from bbox labels using SAM box prompts."""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (
    MASK_BOX_PAD_FRAC,
    MASK_MIN_AREA_BOX_FRAC,
    MASK_MIN_AREA_FLOOR,
    MASK_MIN_BOX_IOU,
    MASK_PREVIEW_COUNT,
    SAM_MODEL,
    SEG_DATASET_ROOT,
)

SRC_ROOT = ROOT / "football_dataset"
PREVIEW_DIR = ROOT / "outputs" / "mask_preview"
QA_DIR = ROOT / "outputs" / "mask_qa"


def yolo_box_to_xyxy(line: str, w: int, h: int) -> tuple[int, int, int, int]:
    parts = line.split()
    cx, cy, bw, bh = map(float, parts[1:5])
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def expand_box(box: tuple[int, int, int, int], w: int, h: int, frac: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * frac)
    pad_y = int(bh * frac)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def min_mask_area(box: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    box_area = max(1, (x2 - x1) * (y2 - y1))
    return int(max(MASK_MIN_AREA_FLOOR, MASK_MIN_AREA_BOX_FRAC * box_area))


def mask_to_polygon(mask: np.ndarray, w: int, h: int) -> list[float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 3:
        return None
    pts = approx.reshape(-1, 2)
    return [coord / dim for pt in pts for coord, dim in zip(pt, [w, h])]


def box_iou(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    box_mask = np.zeros_like(mask)
    box_mask[y1:y2, x1:x2] = 1
    inter = np.logical_and(mask > 0, box_mask > 0).sum()
    union = np.logical_or(mask > 0, box_mask > 0).sum()
    return inter / union if union > 0 else 0.0


def sam_mask_at_index(masks: np.ndarray, idx: int, h: int, w: int) -> np.ndarray:
    m = (masks[idx] > 0.5).astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m * 255).astype(np.uint8)


def validate_mask(m: np.ndarray, box: tuple[int, int, int, int], w: int, h: int) -> tuple[list[float] | None, float, int, str | None]:
    area = int(np.count_nonzero(m))
    iou = box_iou(m, box)
    min_area = min_mask_area(box)
    if area < min_area:
        return None, iou, area, f"area<{min_area}"
    if iou < MASK_MIN_BOX_IOU:
        return None, iou, area, f"iou<{MASK_MIN_BOX_IOU}"
    poly = mask_to_polygon(m, w, h)
    if poly is None or len(poly) < 6:
        return None, iou, area, "bad_polygon"
    return poly, iou, area, None


def build_preview(img: np.ndarray, accepted: list, rejected: list) -> np.ndarray:
    overlay = img.copy()
    for poly, box, _ in accepted:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for box, reason in rejected:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay, (reason or "?")[:12], (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )
    return np.hstack([img, overlay])


def process_split(
    sam: SAM,
    split: str,
    preview_budget: list[int],
    reject_writer: csv.writer,
    stats: dict,
) -> tuple[int, int]:
    img_dir = SRC_ROOT / split / "images"
    lbl_dir = SRC_ROOT / split / "labels"
    out_img = SEG_DATASET_ROOT / split / "images"
    out_lbl = SEG_DATASET_ROOT / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    ok, skip = 0, 0
    for label_path in sorted(lbl_dir.glob("*.txt")):
        stem = label_path.stem
        img_path = img_dir / f"{stem}.jpg"
        if not img_path.exists():
            for ext in (".png", ".jpeg"):
                alt = img_dir / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
            else:
                continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lines = label_path.read_text().strip().splitlines()
        boxes = [yolo_box_to_xyxy(ln, w, h) for ln in lines if ln.strip()]
        if not boxes:
            continue

        seg_lines = []
        accepted_preview = []
        rejected_preview = []
        pending_retry = []

        results = sam(str(img_path), bboxes=boxes, verbose=False)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()
            for bi, box in enumerate(boxes):
                if bi >= len(masks):
                    pending_retry.append((bi, box))
                    continue
                m = sam_mask_at_index(masks, bi, h, w)
                poly, iou, area, reason = validate_mask(m, box, w, h)
                if poly is None:
                    pending_retry.append((bi, box))
                    reject_writer.writerow([split, stem, bi, reason, f"{iou:.3f}", area])
                    rejected_preview.append((box, reason or "fail"))
                    stats["rejected"] += 1
                    skip += 1
                else:
                    seg_lines.append("0 " + " ".join(f"{p:.6f}" for p in poly))
                    ok += 1
                    stats["accepted"] += 1
                    stats["iou_sum"] += iou
                    stats["area_sum"] += area
                    accepted_preview.append((poly, box, iou))
        else:
            pending_retry = [(bi, box) for bi, box in enumerate(boxes)]

        for bi, box in pending_retry:
            padded = expand_box(box, w, h, MASK_BOX_PAD_FRAC)
            r = sam(str(img_path), bboxes=[padded], verbose=False)
            if r[0].masks is None:
                skip += 1
                reject_writer.writerow([split, stem, bi, "retry_no_mask", "0", 0])
                stats["rejected"] += 1
                continue
            m = sam_mask_at_index(r[0].masks.data.cpu().numpy(), 0, h, w)
            poly, iou, area, reason = validate_mask(m, box, w, h)
            if poly is None:
                skip += 1
                reject_writer.writerow([split, stem, bi, f"retry_{reason}", f"{iou:.3f}", area])
                rejected_preview.append((box, reason or "fail"))
                stats["rejected"] += 1
            else:
                seg_lines.append("0 " + " ".join(f"{p:.6f}" for p in poly))
                ok += 1
                stats["accepted"] += 1
                stats["iou_sum"] += iou
                stats["area_sum"] += area
                accepted_preview.append((poly, box, iou))

        if seg_lines:
            shutil.copy2(img_path, out_img / img_path.name)
            (out_lbl / f"{stem}.txt").write_text("\n".join(seg_lines) + "\n")
            if preview_budget[0] > 0:
                preview = build_preview(img, accepted_preview, rejected_preview)
                cv2.imwrite(str(PREVIEW_DIR / f"{split}_{stem}.jpg"), preview)
                preview_budget[0] -= 1

    return ok, skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam", default=SAM_MODEL)
    args = parser.parse_args()

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Run prepare_dataset.py first. Missing {SRC_ROOT}")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    if SEG_DATASET_ROOT.exists():
        shutil.rmtree(SEG_DATASET_ROOT)

    print(f"Loading SAM: {args.sam}")
    sam = SAM(args.sam)

    reject_path = QA_DIR / "rejections.csv"
    stats = {"accepted": 0, "rejected": 0, "iou_sum": 0.0, "area_sum": 0.0}
    preview_budget = [MASK_PREVIEW_COUNT]
    total_ok, total_skip = 0, 0

    with reject_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "stem", "box_idx", "reason", "iou", "area"])
        for split in ("train", "valid"):
            ok, skip = process_split(sam, split, preview_budget, writer, stats)
            total_ok += ok
            total_skip += skip
            print(f"  {split}: {ok} masks written, {skip} rejected")

    total = stats["accepted"] + stats["rejected"]
    keep_rate = stats["accepted"] / total if total else 0.0
    mean_iou = stats["iou_sum"] / stats["accepted"] if stats["accepted"] else 0.0
    mean_area = stats["area_sum"] / stats["accepted"] if stats["accepted"] else 0.0
    summary = {
        "accepted": stats["accepted"],
        "rejected": stats["rejected"],
        "keep_rate": round(keep_rate, 4),
        "mean_iou": round(mean_iou, 4),
        "mean_mask_area": round(mean_area, 1),
        "sam_model": args.sam,
    }
    (QA_DIR / "stats.json").write_text(json.dumps(summary, indent=2))

    print(f"Done. Dataset: {SEG_DATASET_ROOT}")
    print(f"Keep rate: {keep_rate:.1%}  mean IoU: {mean_iou:.3f}")
    print(f"QA: {QA_DIR}/stats.json  previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
