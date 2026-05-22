"""Generate YOLO segmentation labels from bbox labels using SAM box prompts."""

import argparse
import csv
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

from src.config import (
    MASK_BOX_PAD_FRAC,
    MASK_MIN_AREA_BOX_FRAC,
    MASK_MIN_AREA_FLOOR,
    MASK_MIN_BOX_IOU,
    MASK_PREVIEW_COUNT,
    SAM_MODEL,
    SEG_DATASET_ROOT,
)

ROOT = Path(__file__).resolve().parent.parent
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
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    mask_fg = roi > 0
    inter = int(mask_fg.sum())
    box_area = (x2 - x1) * (y2 - y1)
    union = int(mask_fg.size) + box_area - inter
    return inter / union if union > 0 else 0.0


def sam_mask_at_index(masks: np.ndarray, idx: int, h: int, w: int) -> np.ndarray:
    m = (masks[idx] > 0.5).astype(np.uint8)
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return (m * 255).astype(np.uint8)


def validate_mask(
    m: np.ndarray, box: tuple[int, int, int, int], w: int, h: int
) -> tuple[list[float] | None, float, int, str | None]:
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


def build_preview(
    img: np.ndarray, accepted: list, rejected: list, scale: float = 1.0
) -> np.ndarray:
    overlay = img.copy()
    for _poly, box, _ in accepted:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for box, reason in rejected:
        x1, y1, x2, y2 = box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            overlay, (reason or "?")[:12], (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
        )
    combined = np.hstack([img, overlay])
    if scale != 1.0 and scale > 0:
        nh = int(combined.shape[0] * scale)
        nw = int(combined.shape[1] * scale)
        combined = cv2.resize(combined, (nw, nh), interpolation=cv2.INTER_AREA)
    return combined


def _process_retry_batch(
    sam: SAM,
    img: np.ndarray,
    pending_retry: list[tuple[int, tuple]],
    h: int,
    w: int,
) -> list[tuple[int, np.ndarray | None]]:
    if not pending_retry:
        return []
    padded_boxes = [expand_box(box, w, h, MASK_BOX_PAD_FRAC) for _, box in pending_retry]
    results = sam(img, bboxes=padded_boxes, verbose=False)
    out = []
    if results[0].masks is None:
        return [(bi, None) for bi, _ in pending_retry]
    masks = results[0].masks.data.cpu().numpy()
    for i, (bi, _box) in enumerate(pending_retry):
        if i >= len(masks):
            out.append((bi, None))
        else:
            out.append((bi, sam_mask_at_index(masks, i, h, w)))
    return out


def process_split(
    sam: SAM,
    split: str,
    preview_budget: list[int],
    reject_writer: csv.writer,
    stats: dict,
    *,
    src_root: Path,
    out_root: Path,
    resume: bool = False,
    preview_scale: float = 1.0,
    copy_workers: int = 4,
) -> tuple[int, int]:
    img_dir = src_root / split / "images"
    lbl_dir = src_root / split / "labels"
    out_img = out_root / split / "images"
    out_lbl = out_root / split / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    ok, skip = 0, 0
    copy_jobs: list[tuple[Path, Path]] = []

    for label_path in sorted(lbl_dir.glob("*.txt")):
        stem = label_path.stem
        out_label = out_lbl / f"{stem}.txt"
        if resume and out_label.exists():
            continue

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
        pending_retry: list[tuple[int, tuple]] = []

        results = sam(img, bboxes=boxes, verbose=False)
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

        retry_results = _process_retry_batch(sam, img, pending_retry, h, w)
        box_by_idx = {bi: box for bi, box in pending_retry}
        for bi, m in retry_results:
            box = box_by_idx[bi]
            if m is None:
                skip += 1
                reject_writer.writerow([split, stem, bi, "retry_no_mask", "0", 0])
                stats["rejected"] += 1
                continue
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
            copy_jobs.append((img_path, out_img / img_path.name))
            out_label.write_text("\n".join(seg_lines) + "\n")
            if preview_budget[0] > 0:
                preview = build_preview(img, accepted_preview, rejected_preview, preview_scale)
                cv2.imwrite(str(PREVIEW_DIR / f"{split}_{stem}.jpg"), preview)
                preview_budget[0] -= 1

    if copy_jobs:

        def _copy_pair(pair: tuple[Path, Path]) -> None:
            shutil.copy2(pair[0], pair[1])

        with ThreadPoolExecutor(max_workers=copy_workers) as pool:
            list(pool.map(_copy_pair, copy_jobs))

    return ok, skip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam", default=SAM_MODEL)
    parser.add_argument("--resume", action="store_true", help="Skip images with existing labels")
    parser.add_argument("--preview-scale", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4, help="Parallel image copy workers")
    parser.add_argument("--src", default=None, help="Source bbox dataset root (overrides default football_dataset)")
    parser.add_argument("--out", default=None, help="Output seg dataset root (overrides default football_dataset_seg)")
    args = parser.parse_args()

    src_root = Path(args.src).resolve() if args.src else SRC_ROOT
    out_root = Path(args.out).resolve() if args.out else SEG_DATASET_ROOT

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {src_root}")

    # Auto-detect splits by finding subdirs that contain an images/ and labels/ subdir
    splits = sorted(
        d.name for d in src_root.iterdir()
        if d.is_dir() and (d / "images").exists() and (d / "labels").exists()
    )
    if not splits:
        raise FileNotFoundError(f"No valid split dirs found under {src_root}")
    print(f"Detected splits: {splits}")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    QA_DIR.mkdir(parents=True, exist_ok=True)
    if out_root.exists() and not args.resume:
        shutil.rmtree(out_root)

    print(f"Loading SAM: {args.sam}")
    sam = SAM(args.sam)

    reject_path = QA_DIR / "rejections.csv"
    stats = {"accepted": 0, "rejected": 0, "iou_sum": 0.0, "area_sum": 0.0}
    preview_budget = [MASK_PREVIEW_COUNT]
    total_ok, total_skip = 0, 0

    with reject_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "stem", "box_idx", "reason", "iou", "area"])
        for split in splits:
            ok, skip = process_split(
                sam,
                split,
                preview_budget,
                writer,
                stats,
                src_root=src_root,
                out_root=out_root,
                resume=args.resume,
                preview_scale=args.preview_scale,
                copy_workers=args.workers,
            )
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

    print(f"Done. Dataset: {out_root}")
    print(f"Keep rate: {keep_rate:.1%}  mean IoU: {mean_iou:.3f}")
    print(f"QA: {QA_DIR}/stats.json  previews: {PREVIEW_DIR}")


if __name__ == "__main__":
    main()
