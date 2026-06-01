"""Pass 1 IR quality gate to block bad clips and queue mining."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd


def evaluate_ir(
    ir_path: Path,
    *,
    min_median_det_per_frame: float = 12.0,
    max_fp_rate: float = 0.20,
    min_unique_tracks: int = 8,
    max_fallback_rate: float = 0.30,
) -> dict:
    df = pd.read_parquet(ir_path)
    if df.empty:
        return {"pass": False, "reason": "empty_ir"}
    required_cols = {
        "clip_id",
        "frame",
        "track_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "conf",
        "mask_ref",
        "mask_is_fallback",
        "feat_lab6d",
        "torso_px",
        "helmet_hits",
        "is_cut_frame",
        "team",
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        return {"pass": False, "reason": "missing_columns", "missing_columns": missing}
    per_frame = df.groupby("frame")["track_id"].nunique()
    median_det = float(per_frame.median()) if len(per_frame) else 0.0
    unique_tracks = int(df["track_id"].nunique())
    fallback_rate = float(df["mask_is_fallback"].mean()) if "mask_is_fallback" in df.columns else 1.0

    # Simple FP proxy: detections with tiny boxes or extreme aspect.
    w = (df["x2"] - df["x1"]).astype(float).clip(lower=0)
    h = (df["y2"] - df["y1"]).astype(float).clip(lower=1)
    area = w * h
    aspect = w / h
    fp_like = ((area < 900) | (aspect > 2.5)).sum()
    fp_rate = float(fp_like) / float(max(1, len(df)))

    checks = {
        "median_det_per_frame": median_det >= min_median_det_per_frame,
        "fp_rate": fp_rate <= max_fp_rate,
        "unique_tracks": unique_tracks >= min_unique_tracks,
        "fallback_rate": fallback_rate <= max_fallback_rate,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "metrics": {
            "rows": int(len(df)),
            "frames": int(df["frame"].nunique()),
            "median_det_per_frame": median_det,
            "fp_rate": fp_rate,
            "unique_tracks": unique_tracks,
            "fallback_rate": fallback_rate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True)
    parser.add_argument("--report-csv", default=None)
    parser.add_argument("--mine-source", default=None, help="Original clip path for mining")
    parser.add_argument("--auto-mine", action="store_true", help="Run mine_hard_negatives on gate failure")
    parser.add_argument("--mine-limit", type=int, default=80)
    args = parser.parse_args()

    result = evaluate_ir(Path(args.ir))
    print(result)

    if args.report_csv:
        out = Path(args.report_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        metrics = result.get("metrics", {})
        checks = result.get("checks", {})
        row = {"pass": result["pass"], **metrics}
        for k, v in checks.items():
            row[f"check_{k}"] = v
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        print(f"Saved gate report: {out}")

    if not result["pass"] and args.auto_mine and args.mine_source:
        cmd = [
            "python",
            "-m",
            "src.mine_hard_negatives",
            "--source",
            args.mine_source,
            "--split",
            "train",
            "--limit",
            str(args.mine_limit),
            "--min-off-field",
            "1",
        ]
        print("Gate failed. Queueing mining:", " ".join(cmd))
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
