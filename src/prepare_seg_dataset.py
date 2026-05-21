"""Write football_dataset_seg/data.yaml after SAM label generation."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import SEG_DATASET_ROOT


def main() -> None:
    train_imgs = SEG_DATASET_ROOT / "train" / "images"
    val_imgs = SEG_DATASET_ROOT / "valid" / "images"
    if not train_imgs.exists():
        raise FileNotFoundError(
            f"{train_imgs} missing. Run: python src/generate_seg_labels.py"
        )

    n_train = len(list(train_imgs.glob("*")))
    n_val = len(list(val_imgs.glob("*"))) if val_imgs.exists() else 0

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
