"""Prepare helmet detection dataset from NFL Health & Safety CSV labels."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = ROOT / "football_dataset" / "nfl-health-and-safety-helmet-assignment"
RAW_IMAGES = RAW_ROOT / "images"
RAW_LABELS = RAW_ROOT / "image_labels.csv"
OUT_ROOT = ROOT / "football_dataset_helmet"

INCLUDED_LABELS = {"Helmet", "Helmet-Partial"}
OPTIONAL_LABELS = {"Helmet-Blurred", "Helmet-Difficult"}
EXCLUDED_LABELS = {"Helmet-Sideline"}
INCLUDE_OPTIONAL_LABELS = False

TEST_SIZE = 0.1
RANDOM_STATE = 42


def _normalized_box(
    left: float,
    width: float,
    top: float,
    height: float,
    image_w: int,
    image_h: int,
) -> str | None:
    if image_w <= 0 or image_h <= 0 or width <= 0 or height <= 0:
        return None

    x1 = max(0.0, left)
    y1 = max(0.0, top)
    x2 = min(float(image_w), left + width)
    y2 = min(float(image_h), top + height)
    clipped_w = x2 - x1
    clipped_h = y2 - y1
    if clipped_w <= 0 or clipped_h <= 0:
        return None

    cx = ((x1 + x2) / 2.0) / image_w
    cy = ((y1 + y2) / 2.0) / image_h
    w_norm = clipped_w / image_w
    h_norm = clipped_h / image_h
    vals = [cx, cy, w_norm, h_norm]
    if any(v <= 0 for v in vals):
        return None
    if any(v < 0 or v > 1 for v in vals):
        return None
    return f"0 {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f}"


def _load_annotations() -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with RAW_LABELS.open(newline="") as f:
        for row in csv.DictReader(f):
            grouped[row["image"]].append(row)
    if not grouped:
        raise RuntimeError(f"No rows found in {RAW_LABELS}")
    return grouped


def _image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {path}")
    h, w = image.shape[:2]
    return w, h


def _positive_bin(count: int) -> str:
    if count == 0:
        return "none"
    if count <= 2:
        return "few"
    if count <= 6:
        return "mid"
    return "many"


def _gate(train_images: list[str], val_images: list[str], data_yaml: Path) -> None:
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["nc"] == 1, f"Gate fail: nc={cfg['nc']}, expected 1"
    assert cfg["names"] == ["helmet"], f"Gate fail: names={cfg['names']}"
    assert not (set(train_images) & set(val_images)), "Gate fail: train/val overlap"
    print(
        f"Gate PASSED: nc=1, train={len(train_images)}, valid={len(val_images)}, disjoint splits"
    )


def main() -> None:
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(f"Helmet images not found: {RAW_IMAGES}")
    if not RAW_LABELS.exists():
        raise FileNotFoundError(f"Helmet labels not found: {RAW_LABELS}")

    grouped = _load_annotations()
    image_names: list[str] = []
    bins: list[str] = []
    converted: dict[str, list[str]] = {}

    kept_labels = INCLUDED_LABELS | (OPTIONAL_LABELS if INCLUDE_OPTIONAL_LABELS else set())
    skipped_missing = 0

    for image_name in sorted(grouped):
        image_path = RAW_IMAGES / image_name
        if not image_path.exists():
            skipped_missing += 1
            continue

        image_w, image_h = _image_shape(image_path)
        yolo_lines: list[str] = []

        for row in grouped[image_name]:
            label = row["label"]
            if label in EXCLUDED_LABELS or label not in kept_labels:
                continue
            line = _normalized_box(
                left=float(row["left"]),
                width=float(row["width"]),
                top=float(row["top"]),
                height=float(row["height"]),
                image_w=image_w,
                image_h=image_h,
            )
            if line is not None:
                yolo_lines.append(line)

        converted[image_name] = yolo_lines
        image_names.append(image_name)
        bins.append(_positive_bin(len(yolo_lines)))

    if not image_names:
        raise RuntimeError("No helmet images were prepared")

    try:
        train_images, val_images = train_test_split(
            image_names,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=bins,
        )
    except ValueError:
        train_images, val_images = train_test_split(
            image_names,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )

    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split, split_images in (("train", train_images), ("valid", val_images)):
        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for image_name in split_images:
            src_img = RAW_IMAGES / image_name
            dst_img = img_dir / image_name
            shutil.copy2(src_img, dst_img)
            label_path = lbl_dir / f"{Path(image_name).stem}.txt"
            lines = converted[image_name]
            label_path.write_text(("\n".join(lines) + "\n") if lines else "")

    data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["helmet"],
    }
    data_yaml = OUT_ROOT / "data.yaml"
    data_yaml.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    _gate(train_images, val_images, data_yaml)
    print(f"Helmet dataset written to {OUT_ROOT}")
    print(f"  train: {len(train_images)}, valid: {len(val_images)}")
    print(f"  missing images skipped: {skipped_missing}")


if __name__ == "__main__":
    main()
