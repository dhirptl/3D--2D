"""Remap Football Videos-2 labels for single-class player detection eval.

The source dataset has 6 classes:
  0=ball  1=player  2=players  3=referee  4=team_A  5=team_B

Our model predicts one class: 0=Player.

Mapping:
  1 (player), 2 (players), 4 (team_A), 5 (team_B) -> 0
  0 (ball), 3 (referee)                            -> drop

Writes remapped labels to data/football_eval/<split>/labels/ and a
data.yaml that validate_seg.py can consume directly.

Usage:
  python scripts/remap_eval_labels.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.config import ROOT

SRC = ROOT / "Football Videos-2"
DST = ROOT / "data" / "football_eval"

# src class id -> dst class id (None = drop)
REMAP: dict[int, int | None] = {
    0: None,   # ball  -> drop
    1: 0,      # player -> Player
    2: 0,      # players -> Player
    3: None,   # referee -> drop
    4: 0,      # team_A -> Player
    5: 0,      # team_B -> Player
}

# Use test only for cleanest eval; set to ("test", "valid") for 86 images
SPLITS = ("test",)


def remap_label_file(src: Path, dst: Path) -> int:
    """Remap one label file. Returns number of kept boxes."""
    lines = src.read_text().strip().splitlines() if src.exists() else []
    kept = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        cls = int(parts[0])
        mapped = REMAP.get(cls)
        if mapped is not None:
            kept.append(f"{mapped} " + " ".join(parts[1:]))
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(f"Source dataset not found: {SRC}")

    total_images = total_boxes = 0
    for split in SPLITS:
        src_img = SRC / split / "images"
        src_lbl = SRC / split / "labels"
        dst_img = DST / split / "images"
        dst_lbl = DST / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        images = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png"))
        for img in images:
            shutil.copy2(img, dst_img / img.name)
            lbl_src = src_lbl / f"{img.stem}.txt"
            lbl_dst = dst_lbl / f"{img.stem}.txt"
            n = remap_label_file(lbl_src, lbl_dst)
            total_boxes += n
        total_images += len(images)
        print(f"  {split}: {len(images)} images")

    # data.yaml for validate_seg.py
    val_split = SPLITS[0]
    (DST / "data.yaml").write_text(
        f"path: {DST}\n"
        f"train: {SPLITS[-1]}/images\n"
        f"val: {val_split}/images\n"
        + (f"test: test/images\n" if "test" in SPLITS else "")
        + f"nc: 1\n"
        f"names:\n"
        f"- Player\n"
    )

    print(f"\nTotal: {total_images} images, {total_boxes} player boxes")
    print(f"data.yaml: {DST / 'data.yaml'}")
    print()
    print("Score with:")
    print("  .venv/bin/python -m src.validate_seg \\")
    print(f"      --weights football_tracker_seg/run_v3-4_hardneg/weights/best.pt \\")
    print(f"      --data {DST}/data.yaml")


if __name__ == "__main__":
    main()
