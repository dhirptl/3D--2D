"""Fine-tune YOLOv11n on football player dataset (frozen backbone, MPS)."""

import argparse

from ultralytics import YOLO

from src.config import DATASET_YAML
from src.train_common import ROOT, base_train_kwargs


def train(
    batch: int = 8,
    device: str = "mps",
    epochs: int = 100,
    freeze: int = 10,
) -> None:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"{DATASET_YAML} not found. Run: python -m src.prepare_dataset")

    model = YOLO("yolo11n.pt")
    kwargs = base_train_kwargs(
        str(DATASET_YAML),
        str(ROOT / "football_tracker"),
        "run_v1",
        epochs=epochs,
        batch=batch,
        device=device,
        freeze=freeze,
    )
    model.train(**kwargs)
    print(f"Training complete. Weights: {ROOT / 'football_tracker' / 'run_v1' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Frozen backbone layers for early epochs (0 = full fine-tune)",
    )
    args = parser.parse_args()
    train(batch=args.batch, device=args.device, epochs=args.epochs, freeze=args.freeze)
