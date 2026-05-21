"""Fine-tune YOLOv11n-seg on SAM-generated football player masks."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

from src.config import SEG_DATASET_YAML


def train(batch: int = 8, device: str = "mps", name: str = "run_v1") -> None:
    if not SEG_DATASET_YAML.exists():
        raise FileNotFoundError(
            f"{SEG_DATASET_YAML} not found. Run generate_seg_labels.py and prepare_seg_dataset.py"
        )

    model = YOLO("yolo11n-seg.pt")
    model.train(
        data=str(SEG_DATASET_YAML),
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
        overlap_mask=True,
        mask_ratio=4,
        device=device,
        project=str(ROOT / "football_tracker_seg"),
        name=name,
    )
    weights = ROOT / "football_tracker_seg" / name / "weights" / "best.pt"
    print(f"Training complete. Weights: {weights}")
    print("Run: python src/validate_seg.py")


if __name__ == "__main__":
    train()
