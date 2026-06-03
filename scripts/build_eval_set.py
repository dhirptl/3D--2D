"""Build a ground-truth detection eval set from broadcast video.

Samples frames evenly across a clip, runs the player detector to produce
YOLO-format box pre-labels, and writes an images/labels/data.yaml dataset
that an annotator corrects in CVAT/Label Studio. Corrected labels are then
scored with the existing validation tooling.

Pre-labels are a starting point only: the human correction step is what makes
the set ground truth. Box format (not polygon) -> yields box precision/recall/mAP.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    PLAYER_CLASS_ID,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    ROOT,
    SEG_MODEL_PATH,
)

EVAL_ROOT = ROOT / "data" / "broadcast_eval"


def _yolo_box_line(box: np.ndarray, w: int, h: int) -> str:
    """xyxy pixel box -> 'class cx cy bw bh' normalized to [0,1]."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{PLAYER_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def build(
    source: str,
    *,
    out_root: Path = EVAL_ROOT,
    num_frames: int = 150,
    model_path: Path | None = None,
    conf: float = PLAYER_PREDICT_CONF,
    iou: float = PLAYER_PREDICT_IOU,
    imgsz: int = PLAYER_IMGSZ,
    max_det: int = PLAYER_PREDICT_MAX_DET,
) -> None:
    weights = model_path or SEG_MODEL_PATH
    if not weights.exists():
        raise FileNotFoundError(f"Seg model not found: {weights}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        raise RuntimeError(f"Could not read frame count from {source}")

    # Evenly spaced indices span the whole clip -> temporal (≈camera-state) coverage
    # without inventing a shot classifier. Stratify further by hand if needed.
    n = min(num_frames, total)
    indices = np.linspace(0, total - 1, n).astype(int)

    model = YOLO(str(weights))
    img_dir = out_root / "images"
    lbl_dir = out_root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source).stem
    saved = 0
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        h, w = frame.shape[:2]
        results = model.predict(
            frame,
            classes=[PLAYER_CLASS_ID],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        boxes = (
            results[0].boxes.xyxy.cpu().numpy()
            if results and results[0].boxes is not None
            else np.empty((0, 4))
        )
        name = f"{stem}_frame{int(idx):06d}"
        cv2.imwrite(str(img_dir / f"{name}.jpg"), frame)
        lines = [_yolo_box_line(b, w, h) for b in boxes]
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines))
        saved += 1

    cap.release()

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names:\n"
        f"- Player\n"
    )
    print(f"Wrote {saved} frames + pre-labels to {out_root}")
    print(f"data.yaml: {data_yaml}")
    print("Next: correct labels in CVAT/Label Studio, then score with validate_seg.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-root", default=str(EVAL_ROOT))
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=PLAYER_PREDICT_CONF)
    parser.add_argument("--iou", type=float, default=PLAYER_PREDICT_IOU)
    parser.add_argument("--imgsz", type=int, default=PLAYER_IMGSZ)
    parser.add_argument("--max-det", type=int, default=PLAYER_PREDICT_MAX_DET)
    args = parser.parse_args()
    build(
        args.source,
        out_root=Path(args.out_root),
        num_frames=args.num_frames,
        model_path=Path(args.model) if args.model else None,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )


if __name__ == "__main__":
    main()
