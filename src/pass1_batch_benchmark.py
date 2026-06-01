"""Benchmark batched model.predict throughput for Pass 1 optimization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import SEG_MODEL_PATH


def _load_frames(video_path: Path, max_frames: int) -> list:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {video_path}")
    frames = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def benchmark_batches(
    video_path: Path,
    *,
    weights: Path = SEG_MODEL_PATH,
    max_frames: int = 64,
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8),
    imgsz: int = 1280,
) -> dict:
    frames = _load_frames(video_path, max_frames=max_frames)
    if not frames:
        raise RuntimeError("No frames loaded.")
    model = YOLO(str(weights))
    report = {}
    for bs in batch_sizes:
        chunks = [frames[i : i + bs] for i in range(0, len(frames), bs)]
        t0 = time.perf_counter()
        n = 0
        for chunk in chunks:
            model.predict(chunk, imgsz=imgsz, verbose=False)
            n += len(chunk)
        elapsed = time.perf_counter() - t0
        report[str(bs)] = {
            "frames": n,
            "elapsed_sec": elapsed,
            "fps": (n / elapsed) if elapsed > 0 else 0.0,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--weights", default=str(SEG_MODEL_PATH))
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    report = benchmark_batches(
        Path(args.source),
        weights=Path(args.weights),
        max_frames=args.max_frames,
        imgsz=args.imgsz,
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
