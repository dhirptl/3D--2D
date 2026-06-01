"""Evaluate broadcast clip detection quality gates (FP rate + zero-det rate)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    HUD_BOTTOM_PCT,
    HUD_TOP_PCT,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    SEG_MODEL_PATH,
)
from src.post_process import build_field_mask, filter_by_field_area


def _is_hud_box(box: np.ndarray, h: int, w: int) -> bool:
    _ = w
    y1 = float(box[1])
    y2 = float(box[3])
    return y2 <= h * HUD_TOP_PCT or y1 >= h * (1.0 - HUD_BOTTOM_PCT)


def evaluate_clip(
    model: YOLO,
    source: Path,
    *,
    conf: float,
    iou: float,
    imgsz: int,
    max_det: int,
) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    frames = 0
    zero_det = 0
    total_dets = 0
    fp_like = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1
        r = model.predict(
            frame,
            classes=[0],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            verbose=False,
        )
        boxes = np.empty((0, 4), dtype=float)
        confs = np.array([], dtype=float)
        if r and r[0].boxes is not None and len(r[0].boxes):
            boxes = r[0].boxes.xyxy.cpu().numpy()
            confs = r[0].boxes.conf.cpu().numpy()
        if len(boxes) == 0:
            zero_det += 1
            continue

        total_dets += len(boxes)
        h, w = frame.shape[:2]
        field_mask = build_field_mask(frame)
        tids = np.arange(len(boxes))
        keep_boxes, _, keep_ids = filter_by_field_area(frame, boxes, confs, tids, field_mask)
        keep = set(int(i) for i in keep_ids) if len(keep_boxes) else set()
        for i, b in enumerate(boxes):
            area = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
            aspect = (b[2] - b[0]) / max(1.0, (b[3] - b[1]))
            hud = _is_hud_box(b, h, w)
            off_field = i not in keep
            odd_shape = area < 900 or aspect > 2.5
            if hud or off_field or odd_shape:
                fp_like += 1

    cap.release()
    fp_rate = float(fp_like) / float(max(total_dets, 1))
    zero_rate = float(zero_det) / float(max(frames, 1))
    return {
        "source": str(source),
        "frames": frames,
        "detections": total_dets,
        "fp_like_detections": fp_like,
        "fp_rate": fp_rate,
        "zero_det_rate": zero_rate,
        "pass_fp": fp_rate < 0.20,
        "pass_zero": zero_rate < 0.05,
        "pass_all": fp_rate < 0.20 and zero_rate < 0.05,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, help="One or more clips to evaluate")
    parser.add_argument("--model", default=str(SEG_MODEL_PATH))
    parser.add_argument("--conf", type=float, default=PLAYER_PREDICT_CONF)
    parser.add_argument("--iou", type=float, default=PLAYER_PREDICT_IOU)
    parser.add_argument("--imgsz", type=int, default=PLAYER_IMGSZ)
    parser.add_argument("--max-det", type=int, default=PLAYER_PREDICT_MAX_DET)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    model = YOLO(args.model)
    rows = []
    for src in args.sources:
        row = evaluate_clip(
            model,
            Path(src),
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            max_det=args.max_det,
        )
        rows.append(row)
        print(
            f"{src}: fp_rate={row['fp_rate']:.3f} zero_det_rate={row['zero_det_rate']:.3f} "
            f"pass={row['pass_all']}"
        )

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved broadcast metrics to {out}")


if __name__ == "__main__":
    main()
