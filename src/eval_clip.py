"""Evaluate detection + team classification metrics on a video clip."""

import argparse
import csv
import os
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    BOTSORT_CFG,
    BYTETRACK_CFG,
    EVAL_MAX_LABEL_FLIP_RATE,
    EVAL_MAX_LOCKED_FRAME,
    EVAL_MAX_TRACK_ID_CHANGE_RATE,
    EVAL_MAX_ACTIONABLE_ZERO_DET_FRAC,
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
from src.mask_utils import iou_matrix_xyxy
from src.pipeline import VideoPipelineContext, process_video_frame
from src.team_classifier import FootballTeamClassifier
from src.zerodet_metrics import (
    decomposed_rates,
    load_zerodet_jsonl,
    pipeline_buckets,
    adjusted_zero_det_rate,
)

DEFAULT_ANNOTATIONS = ROOT / "annotations" / "sample_labels.csv"
TEAM_IOU_MATCH_THRESHOLD = 0.3


def _score_team_accuracy(
    annotations_path: Path,
    frame_labels: dict[tuple[int, int], int],
    frame_bboxes: dict[tuple[int, int], tuple[int, int, int, int]],
    *,
    team_match: str = "track_id",
) -> dict | None:
    rows: list[dict] = []
    with annotations_path.open() as f:
        for row in csv.DictReader(f):
            rows.append({
                "frame": int(row["frame"]),
                "track_id": int(row["track_id"]),
                "expected": int(row["correct_team"]),
                "bbox": (
                    int(float(row["x1"])),
                    int(float(row["y1"])),
                    int(float(row["x2"])),
                    int(float(row["y2"])),
                )
                if all(k in row for k in ("x1", "y1", "x2", "y2"))
                else None,
            })

    correct = 0
    total = 0
    for ann in rows:
        fid = ann["frame"]
        tid = ann["track_id"]
        expected = ann["expected"]
        if team_match == "iou":
            ann_box = ann.get("bbox")
            if ann_box is None:
                predicted = frame_labels.get((fid, tid), -1)
            else:
                candidates = [
                    (ptid, pbbox, plab)
                    for (pf, ptid), plab in frame_labels.items()
                    if pf == fid and plab in (0, 1)
                    for pbbox in [frame_bboxes.get((pf, ptid))]
                    if pbbox is not None
                ]
                predicted = -1
                if candidates:
                    boxes = np.array([c[1] for c in candidates], dtype=float)
                    ious = iou_matrix_xyxy(
                        np.array([ann_box], dtype=float), boxes
                    )[0]
                    best = int(np.argmax(ious))
                    if ious[best] >= TEAM_IOU_MATCH_THRESHOLD:
                        predicted = candidates[best][2]
        else:
            predicted = frame_labels.get((fid, tid), -1)

        if predicted >= 0:
            total += 1
            correct += int(predicted == expected)

    if total == 0:
        return None
    return {
        "correct": correct,
        "total": total,
        "pct": round(100 * correct / total, 1),
        "match_mode": team_match,
    }


def _release_gate(
    report: dict,
    *,
    team_acc: dict | None,
    require_early_lock: bool,
    gate_mode: str,
    calibration_loaded: bool,
    use_actionable_zero_det: bool = False,
) -> dict[str, bool]:
    ok_dets = report["avg_detections"] >= EVAL_MIN_AVG_DETECTIONS
    zero_key = (
        "actionable_zero_det_rate"
        if use_actionable_zero_det and "actionable_zero_det_rate" in report
        else "zero_det_rate"
    )
    zero_limit = (
        EVAL_MAX_ACTIONABLE_ZERO_DET_FRAC
        if zero_key == "actionable_zero_det_rate"
        else EVAL_MAX_ZERO_DET_FRAC
    )
    ok_zero = report.get(zero_key, report["zero_det_rate"]) <= zero_limit
    ok_locked = (
        report["locked_pct"] >= EVAL_MIN_LOCKED_PCT or report["final_state"] == "LOCKED"
    )
    ok_early = (
        report["locked_frame_index"] is not None
        and report["locked_frame_index"] < EVAL_MAX_LOCKED_FRAME
    ) or (report["final_state"] == "LOCKED" and calibration_loaded)
    ok_flips = report["label_flip_rate"] <= EVAL_MAX_LABEL_FLIP_RATE
    ok_team_acc = team_acc is None or team_acc["pct"] >= EVAL_MIN_TEAM_ACCURACY_PCT
    ok_track_churn = report.get("track_id_change_rate", 0.0) <= EVAL_MAX_TRACK_ID_CHANGE_RATE

    criteria = {
        "detections": ok_dets,
        "zero_det": ok_zero,
        "locked": ok_locked,
        "early_lock": ok_early,
        "label_stability": ok_flips,
        "team_accuracy": ok_team_acc,
        "track_churn": ok_track_churn,
    }
    if not require_early_lock:
        criteria_for_overall = {k: v for k, v in criteria.items() if k != "early_lock"}
    else:
        criteria_for_overall = criteria

    passed = sum(1 for v in criteria_for_overall.values() if v)
    total = len(criteria_for_overall)
    if gate_mode == "relaxed":
        overall = passed >= max(4, total - 2)
    else:
        overall = all(criteria_for_overall.values())

    criteria["overall_strict"] = all(criteria.values())
    criteria["overall_relaxed"] = passed >= max(4, total - 1)
    criteria["overall"] = overall
    return criteria


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
    team_match: str = "track_id",
    debug_filters: bool = False,
    tracker: str = "bytetrack",
    zerodet_jsonl: Path | None = None,
) -> dict:
    zerodet_path = zerodet_jsonl
    if zerodet_path is not None:
        zerodet_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["ZERODET_DEBUG"] = str(zerodet_path)
        zerodet_path.write_text("")

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
    tracker_path = BOTSORT_CFG if tracker == "botsort" else BYTETRACK_CFG
    ctx = VideoPipelineContext(
        model=model,
        classifier=classifier,
        tracker_cfg=tracker_path,
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
        debug_filters=debug_filters,
    )

    calibration_loaded = False
    if calibration_path:
        with open(calibration_path, "rb") as f:
            classifier.load_calibration(pickle.load(f))
        calibration_loaded = True

    state_counts: Counter = Counter()
    frames_with_labeled_tracks = 0
    locked_labeled_tracks = 0
    locked_frames = 0
    label_flips = 0
    prev_locked_label: dict[int, int] = {}
    frame_labels: dict[tuple[int, int], int] = {}
    frame_bboxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}

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

        for det in detections:
            frame_bboxes[(frame_idx, det["track_id"])] = det["bbox"]

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

    ann_path = annotations_path or DEFAULT_ANNOTATIONS
    team_accuracy = None
    if ann_path.exists():
        team_accuracy = _score_team_accuracy(
            ann_path, frame_labels, frame_bboxes, team_match=team_match
        )
        if team_accuracy:
            print(
                f"Team accuracy ({team_accuracy['match_mode']}): "
                f"{team_accuracy['correct']}/{team_accuracy['total']} "
                f"= {team_accuracy['pct']}%"
            )

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
        "helmet_rejected_total": ctx.helmet_rejected_total,
        "mask_warnings": ctx.det_stats.missing_mask_warnings,
        "final_state": classifier.state,
        "warmup_samples": classifier.warmup_count,
        "states": dict(state_counts),
        "team_accuracy": team_accuracy,
    }
    if zerodet_path is not None:
        by_frame = load_zerodet_jsonl(zerodet_path)
        buckets = pipeline_buckets(frame_idx, by_frame)
        report.update(decomposed_rates(buckets, frame_idx))
        report["zero_det_rate_excl_raw0"] = adjusted_zero_det_rate(
            buckets, frame_idx, exclude_raw0=True
        )
        report["zero_det_rate_excl_raw0_sparse"] = adjusted_zero_det_rate(
            buckets, frame_idx, exclude_raw0=True, exclude_sparse=True
        )
        report["zerodet_jsonl"] = str(zerodet_path)
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
        "--team-match",
        choices=("track_id", "iou"),
        default="track_id",
        help="How to match annotations to predictions for team accuracy",
    )
    parser.add_argument(
        "--gate",
        choices=("strict", "relaxed"),
        default="strict",
        help="strict=all criteria; relaxed=pass if most criteria pass",
    )
    parser.add_argument(
        "--require-early-lock",
        action="store_true",
        help="Include early-lock timing in the release gate (default: advisory only)",
    )
    parser.add_argument("--debug-filters", action="store_true")
    parser.add_argument(
        "--tracker",
        choices=("botsort", "bytetrack"),
        default="bytetrack",
        help="Tracker config (default: bytetrack)",
    )
    parser.add_argument(
        "--annotations",
        default=str(DEFAULT_ANNOTATIONS),
        help="CSV with frame,track_id,correct_team for accuracy",
    )
    parser.add_argument(
        "--decomposed-zero-det",
        action="store_true",
        help="Emit ZERODET_DEBUG JSONL and report actionable HUD+formation zero-det buckets",
    )
    parser.add_argument(
        "--zerodet-jsonl",
        default=None,
        help="Path for ZERODET_DEBUG output (default: temp file when --decomposed-zero-det)",
    )
    parser.add_argument(
        "--gate-actionable-zero-det",
        action="store_true",
        help="North-star zero-det gate uses actionable_zero_det_rate (HUD + formation boundary)",
    )
    args = parser.parse_args()

    zerodet_path: Path | None = None
    if args.decomposed_zero_det or args.zerodet_jsonl:
        if args.zerodet_jsonl:
            zerodet_path = Path(args.zerodet_jsonl)
        else:
            fd, tmp = tempfile.mkstemp(suffix=".jsonl")
            os.close(fd)
            zerodet_path = Path(tmp)

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
        team_match=args.team_match,
        debug_filters=args.debug_filters,
        tracker=args.tracker,
        zerodet_jsonl=zerodet_path,
    )

    print("--- Clip Evaluation ---")
    for k, v in report.items():
        print(f"  {k}: {v}")

    criteria = _release_gate(
        report,
        team_acc=report.get("team_accuracy"),
        require_early_lock=args.require_early_lock,
        gate_mode=args.gate,
        calibration_loaded=bool(args.calibration),
        use_actionable_zero_det=args.gate_actionable_zero_det,
    )

    print(f"  Detection target (avg>={EVAL_MIN_AVG_DETECTIONS:.0f}): {'PASS' if criteria['detections'] else 'FAIL'}")
    zero_label = (
        "actionable zero-det"
        if args.gate_actionable_zero_det and "actionable_zero_det_rate" in report
        else "zero-det"
    )
    zero_lim = (
        EVAL_MAX_ACTIONABLE_ZERO_DET_FRAC
        if args.gate_actionable_zero_det and "actionable_zero_det_rate" in report
        else EVAL_MAX_ZERO_DET_FRAC
    )
    print(f"  {zero_label} target (rate<={zero_lim:.0%}): {'PASS' if criteria['zero_det'] else 'FAIL'}")
    if "actionable_zero_det_rate" in report:
        print(f"  Actionable zero-det (HUD+formation): {report['actionable_zero_det_rate']:.1%}")
        if report.get("zero_det_rate_excl_raw0") is not None:
            print(f"  Zero-det excl raw=0: {report['zero_det_rate_excl_raw0']:.1%}")
        if report.get("zero_det_rate_excl_raw0_sparse") is not None:
            print(
                f"  Zero-det excl raw=0+sparse: {report['zero_det_rate_excl_raw0_sparse']:.1%}"
            )
    print(f"  LOCKED target (>={EVAL_MIN_LOCKED_PCT:.0f}% or final LOCKED): {'PASS' if criteria['locked'] else 'FAIL'}")
    print(f"  Label stability target (flip_rate<={EVAL_MAX_LABEL_FLIP_RATE:.2f}): {'PASS' if criteria['label_stability'] else 'FAIL'}")
    print(
        f"  Track churn target (rate<={EVAL_MAX_TRACK_ID_CHANGE_RATE:.2f}): "
        f"{'PASS' if criteria['track_churn'] else 'FAIL'}"
    )
    team_acc = report.get("team_accuracy")
    if team_acc is not None:
        print(
            "  Team accuracy target "
            f"(>={EVAL_MIN_TEAM_ACCURACY_PCT:.0f}%): {'PASS' if criteria['team_accuracy'] else 'FAIL'}"
        )
    if report["locked_frame_index"] is not None:
        advisory = " (advisory)" if not args.require_early_lock else ""
        print(
            f"  Early lock (<{EVAL_MAX_LOCKED_FRAME} frames){advisory}: "
            f"{'PASS' if criteria['early_lock'] else 'FAIL'}"
        )
    print(f"  North-star release gate ({args.gate}): {'PASS' if criteria['overall'] else 'FAIL'}")
    print(f"  Overall strict (all criteria): {'PASS' if criteria['overall_strict'] else 'FAIL'}")
    print(f"  Overall relaxed: {'PASS' if criteria['overall_relaxed'] else 'FAIL'}")


if __name__ == "__main__":
    main()
