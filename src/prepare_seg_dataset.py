"""Write the active seg dataset data.yaml after SAM label generation."""

from pathlib import Path

import yaml

from src.config import SEG_DATASET_ROOT


def _count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    n = 0
    for _ in directory.iterdir():
        n += 1
    return n


def _resolve_val_images() -> tuple[Path, str]:
    """Prefer val/ (merged v2 layout), fall back to valid/ (Roboflow export)."""
    for rel in ("val/images", "valid/images"):
        candidate = SEG_DATASET_ROOT / rel
        if candidate.exists():
            return candidate, rel
    raise FileNotFoundError(
        f"No validation images under {SEG_DATASET_ROOT}/val/images or .../valid/images. "
        "Run: python -m src.generate_seg_labels"
    )


def main() -> None:
    train_imgs = SEG_DATASET_ROOT / "train" / "images"
    if not train_imgs.exists():
        raise FileNotFoundError(
            f"{train_imgs} missing. Run: python -m src.generate_seg_labels"
        )

    val_imgs, val_rel = _resolve_val_images()
    n_train = _count_images(train_imgs)
    n_val = _count_images(val_imgs)

    data = {
        "path": str(SEG_DATASET_ROOT.resolve()),
        "train": "train/images",
        "val": val_rel,
        "nc": 1,
        "names": ["Player"],
    }
    yaml_path = SEG_DATASET_ROOT / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"Wrote {yaml_path}")
    print(f"  train: {n_train}, val ({val_rel}): {n_val}")


if __name__ == "__main__":
    main()
