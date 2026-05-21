"""Fine-tune YOLOv11n-seg on SAM-generated football player masks."""

import argparse

from ultralytics import YOLO

from src.config import SEG_DATASET_YAML
from src.train_common import ROOT, base_train_kwargs


def train(
    batch: int = 8,
    device: str = "mps",
    name: str = "run_v1",
    epochs: int = 100,
    freeze: int = 10,
) -> None:
    if not SEG_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{SEG_DATASET_YAML} not found. Run generate_seg_labels.py and prepare_seg_dataset.py"
        )

    model = YOLO("yolo11n-seg.pt")
    kwargs = base_train_kwargs(
        str(SEG_DATASET_YAML),
        str(ROOT / "football_tracker_seg"),
        name,
        epochs=epochs,
        batch=batch,
        device=device,
        freeze=freeze,
    )
    kwargs["overlap_mask"] = True
    kwargs["mask_ratio"] = 4
    model.train(**kwargs)
    weights = ROOT / "football_tracker_seg" / name / "weights" / "best.pt"
    print(f"Training complete. Weights: {weights}")
    print("Run: python -m src.validate_seg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--name", default="run_v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Frozen backbone layers for early epochs (0 = full fine-tune)",
    )
    args = parser.parse_args()
    train(
        batch=args.batch,
        device=args.device,
        name=args.name,
        epochs=args.epochs,
        freeze=args.freeze,
    )
