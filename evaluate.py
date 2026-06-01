"""Root-level evaluate wrapper: aggregates all metrics into eval_metrics.csv.

Usage:
    python evaluate.py --clip test_clips/eval_clip.mp4

What it measures (maps to the optimization loop acceptance criteria):
    fp_rate           < 0.20   via src.eval_broadcast
    zero_det_rate     < 0.05   via src.eval_clip
    team_label_acc    > 85%    via src.eval_clip + annotations/sample_labels.csv
    fragmentation_rate< 0.15   via src.eval_routes on outputs/eval_detections.csv
    id_switches       < 2      via src.eval_routes on outputs/eval_detections.csv

Route metrics require running video_pipeline.py first to produce the detections CSV.
Team accuracy requires annotations/sample_labels.csv to have correct_team filled in.
"""

from __future__ import annotations

import argparse
import csv
import datetime
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from src.config import (
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    SEG_MODEL_PATH,
)
from src.eval_broadcast import evaluate_clip as broadcast_eval
from src.eval_clip import eval_clip
from src.eval_routes import compute_route_metrics

ANNOTATIONS = Path("annotations/sample_labels.csv")
OUT_CSV_DEFAULT = "eval_metrics.csv"

SCHEMA = [
    "timestamp", "clip",
    # Acceptance criteria
    "fp_rate", "zero_det_rate",
    "team_label_acc_pct", "team_accuracy_correct", "team_accuracy_total",
    "fragmentation_rate", "id_switches",
    # Supporting metrics
    "avg_detections", "locked_pct", "label_flip_rate",
    "track_id_change_rate", "final_state", "warmup_samples",
    # Pass/fail per criterion
    "pass_fp", "pass_zero_det", "pass_team_acc",
    "pass_fragmentation", "pass_id_switches",
    "pass_all",
]


def _has_annotations() -> bool:
    if not ANNOTATIONS.exists():
        return False
    with ANNOTATIONS.open() as f:
        reader = csv.DictReader(f)
        return any(
            row.get("correct_team", "").strip() not in ("", None)
            for row in reader
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate pipeline against all acceptance criteria")
    ap.add_argument("--clip", default="test_clips/eval_clip.mp4", help="Clip to evaluate")
    ap.add_argument(
        "--detections",
        default="outputs/eval_detections.csv",
        help="Detection dump CSV produced by video_pipeline.py (for route metrics)",
    )
    ap.add_argument("--out-csv", default=OUT_CSV_DEFAULT)
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames processed (speeds up eval at cost of coverage)",
    )
    args = ap.parse_args()

    clip = args.clip
    out_csv = Path(args.out_csv)

    # ── 1. Broadcast eval: FP rate ──────────────────────────────────────────
    print(f"[evaluate] Loading model: {SEG_MODEL_PATH}")
    model = YOLO(str(SEG_MODEL_PATH))

    print(f"[evaluate] Broadcast eval on {clip} ...")
    broadcast = broadcast_eval(
        model,
        Path(clip),
        conf=PLAYER_PREDICT_CONF,
        iou=PLAYER_PREDICT_IOU,
        imgsz=PLAYER_IMGSZ,
        max_det=PLAYER_PREDICT_MAX_DET,
    )
    fp_rate = broadcast["fp_rate"]

    # ── 2. Clip eval: detection + team + tracking metrics ───────────────────
    print(f"[evaluate] Clip eval on {clip} ...")
    ann_path = ANNOTATIONS if _has_annotations() else None
    if ann_path is None:
        print("[evaluate] WARNING: annotations/sample_labels.csv has no correct_team values.")
        print("           Team accuracy will not be measured. Fill in correct_team to enable.")

    clip_report = eval_clip(
        clip,
        max_frames=args.max_frames,
        annotations_path=ann_path,
    )
    zero_det_rate = clip_report["zero_det_rate"]
    team_acc = clip_report.get("team_accuracy")

    # ── 3. Route metrics: fragmentation + ID switches ───────────────────────
    fragmentation_rate: float | None = None
    id_switches: float | None = None
    det_csv = Path(args.detections)
    if det_csv.exists():
        print(f"[evaluate] Route metrics from {det_csv} ...")
        df = pd.read_csv(det_csv)
        route = compute_route_metrics(
            df,
            annotations_path=ann_path,
        )
        fragmentation_rate = route["fragmentation_rate"]
        id_switches = route["id_switches"]
    else:
        print(f"[evaluate] No detections CSV at {det_csv}; skipping route metrics.")
        print("           Run: python video_pipeline.py --clip <clip> first.")

    # ── Pass/fail ────────────────────────────────────────────────────────────
    team_acc_pct = team_acc["pct"] if team_acc else None
    pass_fp = fp_rate < 0.20
    pass_zero = zero_det_rate < 0.05
    pass_team = team_acc_pct is not None and team_acc_pct > 85.0
    pass_frag = fragmentation_rate is not None and fragmentation_rate < 0.15
    pass_ids = id_switches is not None and id_switches < 2.0

    measurable = [pass_fp, pass_zero, pass_frag, pass_ids]
    if team_acc_pct is not None:
        measurable.append(pass_team)
    pass_all = all(measurable)

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "clip": clip,
        "fp_rate": round(fp_rate, 4),
        "zero_det_rate": round(zero_det_rate, 4),
        "team_label_acc_pct": team_acc_pct,
        "team_accuracy_correct": team_acc["correct"] if team_acc else None,
        "team_accuracy_total": team_acc["total"] if team_acc else None,
        "fragmentation_rate": round(fragmentation_rate, 4) if fragmentation_rate is not None else None,
        "id_switches": id_switches,
        "avg_detections": clip_report["avg_detections"],
        "locked_pct": clip_report["locked_pct"],
        "label_flip_rate": clip_report["label_flip_rate"],
        "track_id_change_rate": clip_report["track_id_change_rate"],
        "final_state": clip_report["final_state"],
        "warmup_samples": clip_report["warmup_samples"],
        "pass_fp": pass_fp,
        "pass_zero_det": pass_zero,
        "pass_team_acc": pass_team,
        "pass_fragmentation": pass_frag,
        "pass_id_switches": pass_ids,
        "pass_all": pass_all,
    }

    write_header = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCHEMA)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("\n─── Eval Results ──────────────────────────────────────")
    criteria = {
        "fp_rate < 0.20":            (fp_rate,             pass_fp),
        "zero_det_rate < 0.05":      (zero_det_rate,        pass_zero),
        "team_label_acc > 85%":      (team_acc_pct,         pass_team),
        "fragmentation_rate < 0.15": (fragmentation_rate,   pass_frag),
        "id_switches < 2":           (id_switches,          pass_ids),
    }
    for name, (val, ok) in criteria.items():
        status = "PASS" if ok else ("FAIL" if val is not None else "N/A (no data)")
        print(f"  {'✓' if ok else '✗'} {name}: {val}  [{status}]")
    print(f"\n  Overall: {'ALL PASS ✓' if pass_all else 'NOT YET'}")
    print(f"\nAppended to {out_csv}")


if __name__ == "__main__":
    main()
