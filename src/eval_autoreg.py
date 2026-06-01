"""Evaluate automatic field registration coverage on held-out clips."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from src.field_registration import FieldRegistration


def eval_clip(source: Path, *, max_frames: int = 1200) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    reg = FieldRegistration()
    total = 0
    valid = 0
    first_valid = None
    while cap.isOpened() and total < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        reg.update(frame)
        if reg.registration_valid:
            valid += 1
            if first_valid is None:
                first_valid = total - 1
    cap.release()
    valid_pct = (100.0 * valid / total) if total > 0 else 0.0
    return {
        "source": str(source),
        "frames": total,
        "valid_frames": valid,
        "valid_pct": round(valid_pct, 2),
        "first_valid_frame": first_valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True)
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()
    rows = [eval_clip(Path(s), max_frames=args.max_frames) for s in args.sources]
    for row in rows:
        print(row)
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved auto-reg eval to {out}")


if __name__ == "__main__":
    main()
