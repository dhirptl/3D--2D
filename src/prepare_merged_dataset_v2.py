"""Merge American Football + BallGame3 + existing football_dataset into merged_football_dataset_v2.

Preserves original Roboflow train/val/test split assignments to prevent data leakage
(consecutive video frames must not bleed across train and val).

Class mapping:
  American football: class 0 (person) → keep as 0; class 1 (refree) → drop
  BallGame3:         class 0 (player) → keep as 0; class 1 (referee) → drop
  football_dataset:  class 0 (Player) → keep as 0 (already clean)
"""

import random
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
FD = ROOT / "football_dataset"
OUT = ROOT / "merged_football_dataset_v2"

# (source_dir, dest_split) pairs — all Roboflow train/* → train, valid+test → val
SOURCES: list[tuple[Path, str]] = [
    (FD / "American football" / "train", "train"),
    (FD / "American football" / "valid", "val"),
    (FD / "American football" / "test",  "val"),
    (FD / "BallGame3" / "train",          "train"),
    (FD / "BallGame3" / "valid",          "val"),
    (FD / "BallGame3" / "test",           "val"),
    (FD / "train",                         "train"),
    (FD / "valid",                         "val"),
]


def clean_label(label_path: Path) -> list[str]:
    """Return lines keeping only class-0 annotations, remapped to 0."""
    lines = []
    for raw in label_path.read_text().strip().splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        if int(parts[0]) != 0:
            continue
        parts[0] = "0"
        lines.append(" ".join(parts))
    return lines


def find_image(images_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main() -> None:
    random.seed(42)

    # Accumulate entries per output split; deduplicate by stem within each bucket.
    buckets: dict[str, dict[str, tuple[Path, list[str]]]] = {"train": {}, "val": {}}
    skipped_no_image = 0
    skipped_referee_only = 0

    for src_dir, dest_split in SOURCES:
        lbl_dir = src_dir / "labels"
        img_dir = src_dir / "images"
        if not lbl_dir.exists():
            print(f"  [skip] {lbl_dir} not found")
            continue

        bucket = buckets[dest_split]
        added = 0
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            stem = lbl_path.stem
            if stem in bucket:
                continue  # dedup

            img_path = find_image(img_dir, stem)
            if img_path is None:
                skipped_no_image += 1
                continue

            lines = clean_label(lbl_path)
            if not lines:
                skipped_referee_only += 1
                continue

            bucket[stem] = (img_path, lines)
            added += 1

        print(f"  {src_dir.relative_to(ROOT)} → {dest_split}: +{added} images")

    # Shuffle within each bucket and copy to output
    if OUT.exists():
        shutil.rmtree(OUT)

    for split, bucket in buckets.items():
        img_out = OUT / split / "images"
        lbl_out = OUT / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        stems = list(bucket.keys())
        random.shuffle(stems)

        for stem in stems:
            img_path, lines = bucket[stem]
            shutil.copy2(img_path, img_out / img_path.name)
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    # Write data.yaml
    data = {
        "path": str(OUT.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": ["Player"],
    }
    (OUT / "data.yaml").write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    n_train = len(buckets["train"])
    n_val = len(buckets["val"])
    total = n_train + n_val
    print(f"\nDone → {OUT}")
    print(f"  train: {n_train}  val: {n_val}  total: {total}")
    print(f"  skipped (no image): {skipped_no_image}")
    print(f"  skipped (referee-only): {skipped_referee_only}")
    print(f"  data.yaml: {OUT / 'data.yaml'}")


if __name__ == "__main__":
    main()
