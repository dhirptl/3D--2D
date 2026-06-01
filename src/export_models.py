"""Export and benchmark inference models for Phase 7."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import HELMET_MODEL_PATH, POSE_MODEL_PATH, SEG_MODEL_PATH


def _sample_frame(video_path: Path) -> any:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source clip: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame from {video_path}")
    return frame


def _time_predict(model: YOLO, frame, runs: int = 5) -> float:
    t0 = time.perf_counter()
    for _ in range(runs):
        model.predict(frame, verbose=False)
    return (time.perf_counter() - t0) / max(1, runs)


def export_and_benchmark(
    source_clip: Path,
    *,
    seg: Path = SEG_MODEL_PATH,
    pose: Path = POSE_MODEL_PATH,
    helmet: Path = HELMET_MODEL_PATH,
    export_format: str = "coreml",
    out_json: Path | None = None,
) -> dict:
    frame = _sample_frame(source_clip)
    result: dict[str, dict] = {}
    for name, weights in (("seg", seg), ("pose", pose), ("helmet", helmet)):
        if not weights.exists():
            result[name] = {"weights": str(weights), "exists": False}
            continue
        base = YOLO(str(weights))
        base_s = _time_predict(base, frame)
        exported = base.export(format=export_format)
        result[name] = {
            "weights": str(weights),
            "exists": True,
            "baseline_sec_per_frame": base_s,
            "export_format": export_format,
            "export_artifact": str(exported),
        }
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Representative clip for benchmark")
    parser.add_argument("--format", default="coreml", help="Ultralytics export format")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--seg", default=str(SEG_MODEL_PATH))
    parser.add_argument("--pose", default=str(POSE_MODEL_PATH))
    parser.add_argument("--helmet", default=str(HELMET_MODEL_PATH))
    args = parser.parse_args()
    report = export_and_benchmark(
        Path(args.source),
        seg=Path(args.seg),
        pose=Path(args.pose),
        helmet=Path(args.helmet),
        export_format=args.format,
        out_json=Path(args.out_json) if args.out_json else None,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
