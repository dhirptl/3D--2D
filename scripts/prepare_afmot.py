"""Download MOTAF dataset from HuggingFace and convert to YOLO detection format.

Output structure:
  data/motaf/train/images/  data/motaf/train/labels/
  data/motaf/val/images/    data/motaf/val/labels/

Class mapping (matches combined_dataset/data.yaml):
  player   → 0
  referee  → 1
"""

import argparse
import io
import os
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO_ROOT / "data" / "motaf"

# MOTAF class names as they appear in the dataset
_CLASS_MAP = {
    "player": 0,
    "referee": 1,
}


def _yolo_box(bbox, img_w: int, img_h: int) -> str:
    """Convert absolute [x, y, w, h] bbox to YOLO normalised cx cy w h."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _process_split(ds_split, split_name: str) -> None:
    img_dir = OUT_ROOT / split_name / "images"
    lbl_dir = OUT_ROOT / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i, sample in enumerate(ds_split):
        # Image
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_w, img_h = img.size
        img_path = img_dir / f"{i:06d}.jpg"
        img.save(img_path, "JPEG", quality=95)

        # Labels: each sample has a list of annotations
        annotations = sample.get("annotations") or sample.get("objects") or []
        lines = []
        for ann in annotations:
            # Try common field names for class label
            label = (
                ann.get("category")
                or ann.get("label")
                or ann.get("class")
                or ""
            ).lower().strip()
            cls_id = _CLASS_MAP.get(label)
            if cls_id is None:
                continue
            bbox = ann.get("bbox") or ann.get("bounding_box")
            if bbox is None:
                continue
            lines.append(f"{cls_id} {_yolo_box(bbox, img_w, img_h)}")

        lbl_path = lbl_dir / f"{i:06d}.txt"
        lbl_path.write_text("\n".join(lines))

        if (i + 1) % 500 == 0:
            print(f"  [{split_name}] {i+1}/{len(ds_split)} done")

    print(f"[prepare_afmot] {split_name}: {len(ds_split)} samples → {img_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default=None, help="HuggingFace dataset revision/branch")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Run: pip install datasets")

    print("[prepare_afmot] Downloading rinost081/AFMOT …")
    ds = load_dataset("rinost081/AFMOT", revision=args.revision)
    print(f"[prepare_afmot] Splits: {list(ds.keys())}")

    # Map whatever splits exist to train/val
    splits = list(ds.keys())
    train_split = next((s for s in splits if "train" in s.lower()), splits[0])
    val_split = next((s for s in splits if any(k in s.lower() for k in ("val", "valid", "test"))), None)

    _process_split(ds[train_split], "train")
    if val_split and val_split != train_split:
        _process_split(ds[val_split], "val")
    else:
        # No explicit val split — carve off last 10%
        full = ds[train_split]
        n = len(full)
        split_point = int(n * 0.9)
        print(f"[prepare_afmot] No val split found; using last {n - split_point} samples as val")
        # datasets don't support slice directly; iterate manually
        _process_split(full.select(range(split_point)), "train")
        _process_split(full.select(range(split_point, n)), "val")

    print(f"[prepare_afmot] Done. Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
