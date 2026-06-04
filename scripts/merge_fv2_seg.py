"""Merge data/fv2_clean into merged_football_dataset_v2_seg without SAM.

Roboflow FV2 export is mixed format: ~85% already has polygon seg labels; the
rest are bbox-only. SAM box-prompt keep rate on these broadcast bboxes is ~0%
(vs ~2% on the original dataset) due to tight boxes + MASK_MIN_BOX_IOU=0.65.
This script copies polygon labels as-is and converts bbox labels to rectangular
polygons as a fallback.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import ROOT

FV2_CLEAN = ROOT / "data" / "fv2_clean"
SEG_DATASET = ROOT / "merged_football_dataset_v2_seg"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def bbox_line_to_polygon(line: str) -> str | None:
    parts = line.split()
    if len(parts) != 5:
        return None
    cls = parts[0]
    cx, cy, bw, bh = map(float, parts[1:5])
    x1, y1 = _clamp01(cx - bw / 2), _clamp01(cy - bh / 2)
    x2, y2 = _clamp01(cx + bw / 2), _clamp01(cy + bh / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    coords = (x1, y1, x2, y1, x2, y2, x1, y2)
    return f"{cls} " + " ".join(f"{c:.6f}" for c in coords)


def normalize_polygon_line(line: str) -> str | None:
    parts = line.split()
    if len(parts) < 7 or (len(parts) - 1) % 2:
        return None
    cls = parts[0]
    coords = [_clamp01(float(c)) for c in parts[1:]]
    return f"{cls} " + " ".join(f"{c:.6f}" for c in coords)


def convert_label_lines(lines: list[str]) -> tuple[list[str], str]:
    out: list[str] = []
    kind = "empty"
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 5:
            converted = bbox_line_to_polygon(line)
            if converted:
                out.append(converted)
                kind = "bbox" if kind == "empty" else kind
        else:
            converted = normalize_polygon_line(line)
            if converted:
                out.append(converted)
                kind = "polygon" if kind != "bbox" else "mixed"
    return out, kind


def merge_split(split: str, *, dry_run: bool = False) -> dict[str, int]:
    src_img = FV2_CLEAN / split / "images"
    src_lbl = FV2_CLEAN / split / "labels"
    dst_img = SEG_DATASET / ("train" if split == "train" else "val") / "images"
    dst_lbl = SEG_DATASET / ("train" if split == "train" else "val") / "labels"
    stats = {"copied": 0, "skipped": 0, "polygon": 0, "bbox": 0, "empty": 0, "mixed": 0}

    for img_path in sorted(src_img.glob("*.jpg")):
        stem = img_path.stem
        out_img = dst_img / img_path.name
        out_lbl = dst_lbl / f"{stem}.txt"
        if out_img.exists() and out_lbl.exists():
            stats["skipped"] += 1
            continue

        raw_lines = src_lbl.joinpath(f"{stem}.txt").read_text().strip().splitlines()
        seg_lines, kind = convert_label_lines(raw_lines)
        stats[kind if kind in stats else "mixed"] += 1

        if dry_run:
            stats["copied"] += 1
            continue

        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(seg_lines) + ("\n" if seg_lines else ""))
        stats["copied"] += 1

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not FV2_CLEAN.exists():
        raise FileNotFoundError(f"{FV2_CLEAN} not found — run scripts/dedup_fv2.py first")

    total = {"copied": 0, "skipped": 0, "polygon": 0, "bbox": 0, "empty": 0, "mixed": 0}
    for split in ("train", "valid"):
        stats = merge_split(split, dry_run=args.dry_run)
        for k, v in stats.items():
            total[k] += v
        print(f"  {split}: copied={stats['copied']} skipped={stats['skipped']} "
              f"polygon={stats['polygon']} bbox={stats['bbox']} empty={stats['empty']}")

    n_train = len(list((SEG_DATASET / "train" / "images").glob("*.jpg")))
    n_val = len(list((SEG_DATASET / "val" / "images").glob("*.jpg")))
    print(f"\nDataset after merge: train={n_train} val={n_val}")
    print(f"Merged {total['copied']} new images ({total['polygon']} poly, "
          f"{total['bbox']} bbox-rect, {total['empty']} empty)")


if __name__ == "__main__":
    main()
