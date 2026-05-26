"""Fine-tune YOLOv11n on the helmet detection dataset."""

import argparse

from ultralytics import YOLO

from src.train_common import ROOT, base_train_kwargs

HELMET_DATASET_YAML = ROOT / "football_dataset_helmet" / "data.yaml"
HELMET_PROJECT = ROOT / "football_tracker_helmet"
HELMET_RUN_NAME = "run_v1"
HELMET_IMGSZ = 640


def train(
    batch: int = 8,
    device: str = "mps",
    epochs: int = 50,
    freeze: int = 10,
) -> None:
    if not HELMET_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{HELMET_DATASET_YAML} not found. Run: python -m src.prepare_helmet_dataset"
        )

    model = YOLO("yolo11n.pt")
    kwargs = base_train_kwargs(
        str(HELMET_DATASET_YAML),
        str(HELMET_PROJECT),
        HELMET_RUN_NAME,
        epochs=epochs,
        batch=batch,
        device=device,
        freeze=freeze,
    )
    kwargs["imgsz"] = HELMET_IMGSZ
    model.train(**kwargs)
    print(
        "Training complete. Weights: "
        f"{HELMET_PROJECT / HELMET_RUN_NAME / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument(
        "--freeze",
        type=int,
        default=10,
        help="Frozen backbone layers for early epochs (0 = full fine-tune)",
    )
    args = parser.parse_args()
    train(batch=args.batch, device=args.device, epochs=args.epochs, freeze=args.freeze)
