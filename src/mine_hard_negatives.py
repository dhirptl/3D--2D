"""Mine candidate hard-negative frames from video into football_dataset/hard_negatives."""

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
from src.post_process import build_field_mask, filter_by_field_area

HARD_NEG_ROOT = ROOT / "football_dataset" / "hard_negatives"


def _predict_boxes(
    model: YOLO,
    frame: np.ndarray,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> tuple[np.ndarray, np.ndarray]:
    results = model.predict(
        frame,
        classes=[PLAYER_CLASS_ID],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det=max_det,
        verbose=False,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return np.empty((0, 4), dtype=float), np.array([], dtype=float)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    return boxes, confs


def mine(
    source: str,
    *,
    split: str = "train",
    out_root: Path = HARD_NEG_ROOT,
    every: int = 15,
    limit: int = 250,
    min_off_field: int = 2,
    model_path: Path | None = None,
    conf: float = PLAYER_PREDICT_CONF,
    iou: float = PLAYER_PREDICT_IOU,
    imgsz: int = PLAYER_IMGSZ,
    max_det: int = PLAYER_PREDICT_MAX_DET,
) -> None:
    weights = model_path or SEG_MODEL_PATH
    if not weights.exists():
        raise FileNotFoundError(f"Seg model not found: {weights}")

    model = YOLO(str(weights))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    img_dir = out_root / split / "images"
    lbl_dir = out_root / split / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    saved = 0
    stem = Path(source).stem
    while cap.isOpened() and saved < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % max(1, every) != 0:
            frame_idx += 1
            continue

        field_mask = build_field_mask(frame)
        boxes, confs = _predict_boxes(
            model,
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
        )
        if len(boxes) == 0:
            frame_idx += 1
            continue

        tids = np.arange(len(boxes))
        keep_boxes, _, keep_ids = filter_by_field_area(frame, boxes, confs, tids, field_mask)
        keep = {int(idx) for idx in keep_ids} if keep_ids is not None and len(keep_ids) else set()
        off_field = [idx for idx in range(len(boxes)) if idx not in keep]
        if keep or len(off_field) < min_off_field:
            frame_idx += 1
            continue

        out_name = f"{stem}_frame{frame_idx:06d}.jpg"
        cv2.imwrite(str(img_dir / out_name), frame)
        (lbl_dir / f"{Path(out_name).stem}.txt").write_text("")
        saved += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved} hard negatives to {out_root / split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--out-root", default=str(HARD_NEG_ROOT))
    parser.add_argument("--every", type=int, default=15, help="Sample every Nth frame")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--min-off-field", type=int, default=2)
    parser.add_argument("--model", default=None)
    parser.add_argument("--conf", type=float, default=PLAYER_PREDICT_CONF)
    parser.add_argument("--iou", type=float, default=PLAYER_PREDICT_IOU)
    parser.add_argument("--imgsz", type=int, default=PLAYER_IMGSZ)
    parser.add_argument("--max-det", type=int, default=PLAYER_PREDICT_MAX_DET)
    args = parser.parse_args()
    mine(
        args.source,
        split=args.split,
        out_root=Path(args.out_root),
        every=args.every,
        limit=args.limit,
        min_off_field=args.min_off_field,
        model_path=Path(args.model) if args.model else None,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
    )
