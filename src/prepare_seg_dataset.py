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


def main() -> None:
    train_imgs = SEG_DATASET_ROOT / "train" / "images"
    val_imgs = SEG_DATASET_ROOT / "valid" / "images"
    if not train_imgs.exists():
        raise FileNotFoundError(
            f"{train_imgs} missing. Run: python -m src.generate_seg_labels"
        )

    n_train = _count_images(train_imgs)
    n_val = _count_images(val_imgs)

    data = {
        "path": str(SEG_DATASET_ROOT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["Player"],
    }
    yaml_path = SEG_DATASET_ROOT / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    print(f"Wrote {yaml_path}")
    print(f"  train: {n_train}, valid: {n_val}")


if __name__ == "__main__":
    main()
