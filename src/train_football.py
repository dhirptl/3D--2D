"""Fine-tune YOLOv11n on football player dataset (frozen backbone, MPS)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.config import DATASET_YAML


def train(batch: int = 8, device: str = "mps") -> None:
    data_yaml = DATASET_YAML
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"{data_yaml} not found. Run: python src/prepare_dataset.py"
        )

    model = YOLO("yolo11n.pt")
    results = model.train(
        data=str(data_yaml),
        epochs=150,
        imgsz=736,
        batch=batch,
        patience=30,
        freeze=10,
        mosaic=1.0,
        mixup=0.15,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        scale=0.5,
        translate=0.1,
        device=device,
        project=str(ROOT / "football_tracker"),
        name="run_v1",
    )
    print(f"Training complete. Weights: {ROOT / 'football_tracker' / 'run_v1' / 'weights' / 'best.pt'}")
    return results


if __name__ == "__main__":
    train()
