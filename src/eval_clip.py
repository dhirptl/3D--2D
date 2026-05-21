"""Evaluate detection + team classification metrics on a video clip."""

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import QUALITY_FRAMES, SEG_MODEL_PATH
from src.detection import DetectionStats, extract_tracked_detections
from src.field_hsv import build_field_mask_from_bounds, estimate_field_hsv
from src.team_classifier import FootballTeamClassifier
from ultralytics import YOLO


def eval_clip(source: str, model_path: Path | None = None, max_frames: int | None = None) -> dict:
    weights = model_path or SEG_MODEL_PATH
    model = YOLO(str(weights))
    classifier = FootballTeamClassifier()
    det_stats = DetectionStats()
    state_counts: Counter = Counter()
    team_label_counts: Counter = Counter()

    cap = cv2.VideoCapture(source)
    auto_frames = []
    field_hsv_bounds = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        if field_hsv_bounds is None:
            auto_frames.append(frame)
            if len(auto_frames) >= 30:
                field_hsv_bounds = estimate_field_hsv(auto_frames)
                classifier.field_hsv = field_hsv_bounds
                classifier.field_mask = build_field_mask_from_bounds(frame, *field_hsv_bounds)

        if field_hsv_bounds is not None:
            classifier.field_mask = build_field_mask_from_bounds(frame, *field_hsv_bounds)

        detections = extract_tracked_detections(
            model, frame, frame_idx=frame_idx, stats=det_stats
        )
        labels = classifier.process_frame(frame, detections)
        state_counts[classifier.state] += 1
        for tid, lab in labels.items():
            team_label_counts[lab] += 1

        frame_idx += 1

    cap.release()

    avg_dets = det_stats.total_dets / det_stats.frames if det_stats.frames else 0
    locked_pct = (
        100 * state_counts.get(FootballTeamClassifier.STATE_LOCKED, 0) / frame_idx
        if frame_idx else 0
    )
    labeled_pct = (
        100 * (team_label_counts.get(0, 0) + team_label_counts.get(1, 0))
        / max(1, sum(team_label_counts.values()))
    )

    report = {
        "frames": frame_idx,
        "avg_detections": round(avg_dets, 2),
        "max_detections": det_stats.max_dets,
        "zero_det_frames": det_stats.zero_det_frames,
        "locked_pct": round(locked_pct, 1),
        "team_labeled_pct": round(labeled_pct, 1),
        "final_state": classifier.state,
        "warmup_samples": classifier.warmup_count,
        "states": dict(state_counts),
        "team_labels": dict(team_label_counts),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    report = eval_clip(args.source, Path(args.model) if args.model else None, args.max_frames)

    print("--- Clip Evaluation ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    ok_dets = report["avg_detections"] >= 18
    ok_locked = report["locked_pct"] >= 80 or report["final_state"] == "LOCKED"
    print(f"  Detection target (avg>=18): {'PASS' if ok_dets else 'FAIL'}")
    print(f"  LOCKED target (>80% or final LOCKED): {'PASS' if ok_locked else 'FAIL'}")


if __name__ == "__main__":
    main()
