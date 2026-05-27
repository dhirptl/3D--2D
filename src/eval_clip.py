"""Evaluate detection + team classification metrics on a video clip."""

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.config import (
    EVAL_MAX_LABEL_FLIP_RATE,
    EVAL_MAX_LOCKED_FRAME,
    EVAL_MAX_ZERO_DET_FRAC,
    EVAL_MIN_AVG_DETECTIONS,
    EVAL_MIN_LOCKED_PCT,
    EVAL_MIN_TEAM_ACCURACY_PCT,
    HELMET_CONF,
    HELMET_EVERY_DEFAULT,
    HELMET_GATE_GRACE,
    HELMET_MODEL_PATH,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    ROOT,
    SEG_MODEL_PATH,
)
from src.pipeline import VideoPipelineContext, process_video_frame
from src.team_classifier import FootballTeamClassifier

DEFAULT_ANNOTATIONS = ROOT / "annotations" / "sample_labels.csv"


def eval_clip(
    source: str,
    model_path: Path | None = None,
    max_frames: int | None = None,
    calibration_path: str | None = None,
    annotations_path: Path | None = None,
    player_conf: float = PLAYER_PREDICT_CONF,
    player_iou: float = PLAYER_PREDICT_IOU,
    player_imgsz: int = PLAYER_IMGSZ,
    player_max_det: int = PLAYER_PREDICT_MAX_DET,
    require_helmet: bool = True,
    helmet_model_path: Path | None = None,
    helmet_conf: float = HELMET_CONF,
    helmet_every: int = HELMET_EVERY_DEFAULT,
    use_helmet_gate: bool = True,
    helmet_gate_grace: int = HELMET_GATE_GRACE,
) -> dict:
    weights = model_path or SEG_MODEL_PATH
    if not weights.exists():
        raise FileNotFoundError(f"Seg model not found: {weights}")
    helmet_weights = helmet_model_path or HELMET_MODEL_PATH
    if require_helmet and not helmet_weights.exists():
        raise FileNotFoundError(
            "Helmet model not found: "
            f"{helmet_weights}. Run: python -m src.prepare_helmet_dataset "
            "&& python -m src.train_helmet_detector"
        )

    model = YOLO(str(weights))
    classifier = FootballTeamClassifier()
    ctx = VideoPipelineContext(
        model=model,
        classifier=classifier,
        player_conf=player_conf,
        player_iou=player_iou,
        player_imgsz=max(64, int(player_imgsz)),
        player_max_det=max(1, int(player_max_det)),
        require_helmet=require_helmet,
        helmet_model_path=helmet_weights,
        helmet_conf=helmet_conf,
        helmet_every=max(1, int(helmet_every)),
        use_helmet_gate=use_helmet_gate,
        helmet_gate_grace=max(0, helmet_gate_grace),
    )

    if calibration_path:
        with open(calibration_path, "rb") as f:
            classifier.load_calibration(pickle.load(f))

    state_counts: Counter = Counter()
    frames_with_labeled_tracks = 0
    locked_labeled_tracks = 0
    locked_frames = 0
    label_flips = 0
    prev_locked_label: dict[int, int] = {}
    frame_labels: dict[tuple[int, int], int] = {}

    cap = cv2.VideoCapture(source)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        detections, team_labels = process_video_frame(frame, ctx)
        state_counts[classifier.state] += 1

        if team_labels:
            for tid, lab in team_labels.items():
                if lab in (0, 1):
                    frame_labels[(frame_idx, tid)] = lab

        if classifier.state == FootballTeamClassifier.STATE_LOCKED:
            locked_frames += 1
            if team_labels:
                labeled_ids = {tid for tid, lab in team_labels.items() if lab in (0, 1)}
                if labeled_ids:
                    frames_with_labeled_tracks += 1
                    locked_labeled_tracks += len(labeled_ids)
                for tid, lab in (team_labels or {}).items():
                    if lab not in (0, 1):
                        continue
                    if tid in classifier.voter.locked_team:
                        prev = prev_locked_label.get(tid)
                        if prev is not None and prev != lab:
                            label_flips += 1
                        prev_locked_label[tid] = lab
        elif team_labels:
            labeled_ids = {tid for tid, lab in team_labels.items() if lab in (0, 1)}
            if labeled_ids:
                frames_with_labeled_tracks += 1

        frame_idx += 1

    cap.release()

    team_accuracy: dict | None = None
    ann_path = annotations_path or DEFAULT_ANNOTATIONS
    if ann_path.exists():
        correct = 0
        total = 0
        with ann_path.open() as f:
            for row in csv.DictReader(f):
                fid = int(row["frame"])
                tid = int(row["track_id"])
                expected = int(row["correct_team"])
                predicted = frame_labels.get((fid, tid), -1)
                if predicted >= 0:
                    total += 1
                    correct += int(predicted == expected)
        if total > 0:
            team_accuracy = {
                "correct": correct,
                "total": total,
                "pct": round(100 * correct / total, 1),
            }
            print(f"Team accuracy: {correct}/{total} = {team_accuracy['pct']}%")

    avg_dets = ctx.det_stats.total_dets / ctx.det_stats.frames if ctx.det_stats.frames else 0
    locked_pct = (
        100 * state_counts.get(FootballTeamClassifier.STATE_LOCKED, 0) / frame_idx
        if frame_idx else 0
    )
    labeled_pct = (
        100 * frames_with_labeled_tracks / frame_idx if frame_idx else 0
    )
    avg_locked_labels = (
        locked_labeled_tracks / locked_frames if locked_frames else 0
    )
    flip_rate = label_flips / max(1, len(prev_locked_label))

    locked_at = classifier.locked_frame_index
    if locked_at is not None and locked_at < 0:
        locked_at = None

    reg_pct = (
        100 * ctx.registration_valid_frames / frame_idx if frame_idx else 0
    )
    id_change_rate = (
        round(ctx.track_id_changes / max(1, frame_idx), 4)
    )

    report = {
        "frames": frame_idx,
        "avg_detections": round(avg_dets, 2),
        "max_detections": ctx.det_stats.max_dets,
        "zero_det_frames": ctx.det_stats.zero_det_frames,
        "zero_det_rate": round(ctx.det_stats.zero_det_frames / frame_idx, 4) if frame_idx else 0.0,
        "locked_pct": round(locked_pct, 1),
        "frames_with_team_labels_pct": round(labeled_pct, 1),
        "locked_frame_index": locked_at,
        "avg_labeled_tracks_per_locked_frame": round(avg_locked_labels, 2),
        "label_flip_rate": round(flip_rate, 4),
        "registration_valid_pct": round(reg_pct, 1),
        "track_id_change_rate": id_change_rate,
        "final_state": classifier.state,
        "warmup_samples": classifier.warmup_count,
        "states": dict(state_counts),
        "team_accuracy": team_accuracy,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--calibration", default=None, help="Load saved calibration pickle")
    parser.add_argument("--player-conf", type=float, default=PLAYER_PREDICT_CONF)
    parser.add_argument("--player-iou", type=float, default=PLAYER_PREDICT_IOU)
    parser.add_argument("--player-imgsz", type=int, default=PLAYER_IMGSZ)
    parser.add_argument("--player-max-det", type=int, default=PLAYER_PREDICT_MAX_DET)
    parser.add_argument("--no-helmet", action="store_true")
    parser.add_argument("--helmet-model", default=None)
    parser.add_argument("--helmet-conf", type=float, default=HELMET_CONF)
    parser.add_argument("--helmet-every", type=int, default=HELMET_EVERY_DEFAULT)
    parser.add_argument("--no-helmet-gate", action="store_true")
    parser.add_argument("--helmet-gate-grace", type=int, default=HELMET_GATE_GRACE)
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
        help="CSV with frame,track_id,correct_team for accuracy",
    )
    args = parser.parse_args()

    report = eval_clip(
        args.source,
        Path(args.model) if args.model else None,
        args.max_frames,
        args.calibration,
        Path(args.annotations) if args.annotations else None,
        args.player_conf,
        args.player_iou,
        args.player_imgsz,
        args.player_max_det,
        not args.no_helmet,
        Path(args.helmet_model) if args.helmet_model else None,
        args.helmet_conf,
        args.helmet_every,
        not args.no_helmet_gate,
        args.helmet_gate_grace,
    )

    print("--- Clip Evaluation ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    ok_dets = report["avg_detections"] >= EVAL_MIN_AVG_DETECTIONS
    ok_zero = report["zero_det_rate"] <= EVAL_MAX_ZERO_DET_FRAC
    ok_locked = report["locked_pct"] >= EVAL_MIN_LOCKED_PCT or report["final_state"] == "LOCKED"
    ok_early = (
        report["locked_frame_index"] is not None
        and report["locked_frame_index"] < EVAL_MAX_LOCKED_FRAME
    ) or report["final_state"] == "LOCKED" and args.calibration
    ok_flips = report["label_flip_rate"] <= EVAL_MAX_LABEL_FLIP_RATE
    team_acc = report.get("team_accuracy")
    ok_team_acc = team_acc is None or team_acc["pct"] >= EVAL_MIN_TEAM_ACCURACY_PCT

    print(f"  Detection target (avg>={EVAL_MIN_AVG_DETECTIONS:.0f}): {'PASS' if ok_dets else 'FAIL'}")
    print(f"  Zero-det target (rate<={EVAL_MAX_ZERO_DET_FRAC:.0%}): {'PASS' if ok_zero else 'FAIL'}")
    print(f"  LOCKED target (>={EVAL_MIN_LOCKED_PCT:.0f}% or final LOCKED): {'PASS' if ok_locked else 'FAIL'}")
    print(f"  Label stability target (flip_rate<={EVAL_MAX_LABEL_FLIP_RATE:.2f}): {'PASS' if ok_flips else 'FAIL'}")
    if team_acc is not None:
        print(
            "  Team accuracy target "
            f"(>={EVAL_MIN_TEAM_ACCURACY_PCT:.0f}%): {'PASS' if ok_team_acc else 'FAIL'}"
        )
    if report["locked_frame_index"] is not None:
        print(f"  Early lock (<{EVAL_MAX_LOCKED_FRAME} frames): {'PASS' if ok_early else 'FAIL'}")

    overall = ok_dets and ok_zero and ok_locked and ok_early and ok_flips and ok_team_acc
    print(f"  North-star release gate: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
