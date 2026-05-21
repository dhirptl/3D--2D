"""End-to-end seg detect + track + team classification."""

import argparse
import csv
import pickle
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_PATH, ROOT as CONFIG_ROOT, SEG_MODEL_PATH
from src.detection import DetectionStats, extract_tracked_detections
from src.draw import draw_teams
from src.field_hsv import build_field_mask_from_bounds, estimate_field_hsv
from src.post_process import build_field_mask, filter_by_field_area
from src.team_classifier import FootballTeamClassifier


def load_calibration(classifier: FootballTeamClassifier, path: Path) -> None:
    with open(path, "rb") as f:
        data = pickle.load(f)
    classifier.load_calibration(data["scaler"], data["centroid_team0"], data["centroid_team1"])
    print(f"Loaded calibration from {path}")


def run(
    source: str,
    out: str,
    model_path: Path | None = None,
    *,
    use_team: bool = True,
    filter_field: bool = False,
    apply_hud_filter: bool = True,
    show_masks: bool = False,
    save_calibration: str | None = None,
    load_calibration_path: str | None = None,
    dump_detections: str | None = None,
) -> None:
    weights = model_path or (SEG_MODEL_PATH if SEG_MODEL_PATH.exists() else MODEL_PATH)
    if not weights.exists():
        raise FileNotFoundError(
            f"Model not found: {weights}. Train seg model: python src/train_football_seg.py"
        )

    model = YOLO(str(weights))
    classifier = FootballTeamClassifier() if use_team else None
    det_stats = DetectionStats()
    team_state_counts: Counter = Counter()
    labeled_frames = 0
    total_frames = 0

    if load_calibration_path and classifier is not None:
        load_calibration(classifier, Path(load_calibration_path))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )

    dump_rows = []
    auto_frames = []
    field_hsv_bounds = None

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total_frames += 1

        if classifier is not None and field_hsv_bounds is None:
            auto_frames.append(frame)
            if len(auto_frames) >= 30:
                field_hsv_bounds = estimate_field_hsv(auto_frames)
                classifier.field_hsv = field_hsv_bounds
                print(f"Auto field HSV: {field_hsv_bounds}")

        field_mask = build_field_mask(frame)
        if classifier is not None:
            if field_hsv_bounds is not None:
                classifier.field_mask = build_field_mask_from_bounds(frame, *field_hsv_bounds)
            else:
                classifier.field_mask = field_mask

        detections = extract_tracked_detections(
            model,
            frame,
            frame_idx=frame_idx,
            apply_hud_filter=apply_hud_filter,
            stats=det_stats,
        )

        if filter_field and detections:
            xyxy = np.array([d["bbox"] for d in detections], dtype=float)
            confs = np.array([d["conf"] for d in detections])
            tids = np.array([d["track_id"] for d in detections])
            fm = classifier.field_mask if classifier and classifier.field_mask is not None else field_mask
            xyxy, confs, tids = filter_by_field_area(frame, xyxy, confs, tids, fm)
            keep = {int(t) for t in tids} if len(tids) else set()
            detections = [d for d in detections if d["track_id"] in keep]

        if classifier is not None:
            team_labels = classifier.process_frame(frame, detections)
            team_state_counts[classifier.state] += 1
            if classifier.state == FootballTeamClassifier.STATE_LOCKED:
                labeled_frames += 1
            annotated = draw_teams(
                frame, detections, team_labels, classifier, show_masks=show_masks
            )
        else:
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if dump_detections:
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                dump_rows.append([
                    frame_idx,
                    det["track_id"],
                    det["conf"],
                    x2 - x1,
                    y2 - y1,
                    np.count_nonzero(det["mask"]),
                    classifier.state if classifier else "none",
                ])

        writer.write(annotated)
        frame_idx += 1

    cap.release()
    writer.release()

    if dump_detections and dump_rows:
        dp = Path(dump_detections)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with dp.open("w", newline="") as f:
            wtr = csv.writer(f)
            wtr.writerow(["frame", "track_id", "conf", "box_w", "box_h", "mask_area", "classifier_state"])
            wtr.writerows(dump_rows)

    if save_calibration and classifier is not None and classifier.state == FootballTeamClassifier.STATE_LOCKED:
        cal_path = Path(save_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "wb") as f:
            pickle.dump(
                {
                    "scaler": classifier.scaler,
                    "centroid_team0": classifier.centroid_team0,
                    "centroid_team1": classifier.centroid_team1,
                },
                f,
            )
        print(f"Saved calibration to {cal_path}")

    print(det_stats.summary())
    print(f"Wrote {frame_idx} frames to {out_path}")
    if classifier is not None:
        print(f"Classifier states: {dict(team_state_counts)}")
        print(f"Final state: {classifier.state}  warmup_samples: {classifier.warmup_count}")
        if total_frames:
            locked_pct = 100 * team_state_counts.get(FootballTeamClassifier.STATE_LOCKED, 0) / total_frames
            print(f"LOCKED frames: {locked_pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Football seg + track + teams")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(CONFIG_ROOT / "outputs" / "tracked.mp4"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-team", action="store_true")
    parser.add_argument("--filter-field", action="store_true")
    parser.add_argument("--no-hud-filter", action="store_true")
    parser.add_argument("--show-masks", action="store_true")
    parser.add_argument("--save-calibration", default=None)
    parser.add_argument("--load-calibration", default=None)
    parser.add_argument("--dump-detections", default=None)
    args = parser.parse_args()

    run(
        args.source,
        args.out,
        Path(args.model) if args.model else None,
        use_team=not args.no_team,
        filter_field=args.filter_field,
        apply_hud_filter=not args.no_hud_filter,
        show_masks=args.show_masks,
        save_calibration=args.save_calibration,
        load_calibration_path=args.load_calibration,
        dump_detections=args.dump_detections,
    )


if __name__ == "__main__":
    main()
