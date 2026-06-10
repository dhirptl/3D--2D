"""Filter Football Videos-2 train+valid to images not already in the seg training set.

Writes clean (non-duplicate) images + remapped single-class labels to
data/fv2_clean/{train,valid}/{images,labels}/ — ready for SAM mask generation.

Usage:
  python scripts/dedup_fv2.py
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from src.config import ROOT

FV2_ROOT = ROOT / "Football Videos-2"
EX_ROOTS = [
    ROOT / "merged_football_dataset_v2_seg" / "train" / "images",
    ROOT / "merged_football_dataset_v2_seg" / "val" / "images",
]
OUT_ROOT = ROOT / "data" / "fv2_clean"

# Remap Football Videos-2 classes to single Player class (0)
# 0=ball -> drop, 1=player -> 0, 2=players -> 0,
# 3=referee -> drop, 4=team_A -> 0, 5=team_B -> 0
REMAP: dict[int, int | None] = {0: None, 1: 0, 2: 0, 3: None, 4: 0, 5: 0}


def _md5(p: Path) -> str:
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def _remap_label(src: Path, dst: Path) -> int:
    lines = src.read_text().strip().splitlines() if src.exists() else []
    kept = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        mapped = REMAP.get(int(parts[0]))
        if mapped is not None:
            kept.append(f"{mapped} " + " ".join(parts[1:]))
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def main() -> None:
    # Build hash set of existing training images
    print("Building hash index of existing training set ...")
    existing: set[str] = set()
    for d in EX_ROOTS:
        for img in d.glob("*.jpg"):
            existing.add(_md5(img))
    print(f"  {len(existing)} unique images in current training set")

    total = kept = skipped = 0
    for split in ("train", "valid"):
        src_img_dir = FV2_ROOT / split / "images"
        src_lbl_dir = FV2_ROOT / split / "labels"
        out_img_dir = OUT_ROOT / split / "images"
        out_lbl_dir = OUT_ROOT / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        split_kept = split_skip = 0
        for img in sorted(src_img_dir.glob("*.jpg")):
            total += 1
            if _md5(img) in existing:
                skipped += 1
                split_skip += 1
                continue
            shutil.copy2(img, out_img_dir / img.name)
            _remap_label(src_lbl_dir / f"{img.stem}.txt", out_lbl_dir / f"{img.stem}.txt")
            kept += 1
            split_kept += 1

        print(f"  {split}: {split_kept} kept, {split_skip} duplicates dropped")

    print(f"\nTotal: {kept}/{total} images written to {OUT_ROOT}")
    print(f"Duplicates dropped: {skipped} ({100*skipped//total}%)")
    print()
    print("Next: generate SAM masks and merge into training set:")
    print(f"  .venv/bin/python src/generate_seg_labels.py \\")
    print(f"      --src {OUT_ROOT}/train \\")
    print(f"      --out merged_football_dataset_v2_seg")


if __name__ == "__main__":
    main()
