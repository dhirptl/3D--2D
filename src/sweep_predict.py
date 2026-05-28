"""Sweep player detection conf/iou and report avg detections per frame."""

import argparse
import csv
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import (
    PLAYER_CLASS_ID,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    SEG_MODEL_PATH,
)


def sweep(
    source: str,
    model_path: Path,
    *,
    conf_values: list[float],
    iou_values: list[float],
    max_frames: int | None = None,
    out_csv: Path | None = None,
) -> list[dict]:
    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(source)
    rows: list[dict] = []

    for conf in conf_values:
        for iou in iou_values:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            total = 0
            frames = 0
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break
                results = model.predict(
                    frame,
                    classes=[PLAYER_CLASS_ID],
                    conf=conf,
                    iou=iou,
                    imgsz=PLAYER_IMGSZ,
                    max_det=PLAYER_PREDICT_MAX_DET,
                    verbose=False,
                )
                n = len(results[0].boxes) if results[0].boxes is not None else 0
                total += n
                frames += 1
                frame_idx += 1
            avg = total / frames if frames else 0.0
            row = {"conf": conf, "iou": iou, "avg_detections": round(avg, 2), "frames": frames}
            rows.append(row)
            print(f"conf={conf:.2f} iou={iou:.2f} -> avg_detections={avg:.2f}")

    cap.release()
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["conf", "iou", "avg_detections", "frames"])
            w.writeheader()
            w.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=str(SEG_MODEL_PATH))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out", default="outputs/sweep_predict.csv")
    parser.add_argument(
        "--conf",
        default="0.20,0.25,0.30,0.35",
        help="Comma-separated conf values",
    )
    parser.add_argument(
        "--iou",
        default="0.50,0.65,0.75",
        help="Comma-separated iou values",
    )
    args = parser.parse_args()
    conf_values = [float(x) for x in args.conf.split(",")]
    iou_values = [float(x) for x in args.iou.split(",")]
    sweep(
        args.source,
        Path(args.model),
        conf_values=conf_values,
        iou_values=iou_values,
        max_frames=args.max_frames,
        out_csv=Path(args.out),
    )


if __name__ == "__main__":
    main()
