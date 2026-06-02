"""Pass 2: analysis-only stage over Pass 1 IR.

Team labels use clip-level offline clustering (see offline_teams). Route boundaries
use the first detected snap only; multi-play clips are not split into per-play
routes until multi-snap detection is added and validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.detect_snap import detect_snap_frame, trim_routes
from src.eval_routes import compute_route_metrics
from src.field_registration import FieldRegistration
from src.jersey_ocr import aggregate_numbers
from src.offline_teams import assign_teams_offline, resolve_source_clip
from src.stitch_tracklets import stitch_tracklets


def _project_foot(h: np.ndarray, row: pd.Series) -> tuple[float, float]:
    x = (float(row["x1"]) + float(row["x2"])) / 2.0
    y = float(row["y2"])
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = h @ p
    if abs(q[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(q[0] / q[2]), float(q[1] / q[2])


def run_pass2(
    ir_path: Path,
    out_dir: Path,
    *,
    homography_json: Path | None = None,
    source_clip: Path | None = None,
    annotations_csv: Path | None = None,
    max_gap: int = 30,
    max_dist_px: float = 120.0,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(ir_path)
    if df.empty:
        raise RuntimeError("IR is empty.")

    video = resolve_source_clip(source_clip, ir_path)
    df = assign_teams_offline(df, source_clip=video, ir_path=ir_path)
    df = stitch_tracklets(df, max_gap=max_gap, max_dist_px=max_dist_px)

    # Re-vote teams on stable IDs after stitching.
    if "team" in df.columns and "stable_id" in df.columns:
        voted = (
            df[df["team"] >= 0]
            .groupby("stable_id")["team"]
            .agg(lambda s: int(s.mode().iloc[0]) if len(s.mode()) else -1)
            .to_dict()
        )
        df["team"] = df["stable_id"].map(lambda sid: voted.get(sid, -1))
    if "ocr_num" in df.columns and "stable_id" in df.columns:
        num_map = aggregate_numbers(df, id_col="stable_id", min_conf=0.5)
        df["jersey_number"] = df["stable_id"].map(lambda sid: num_map.get(int(sid)))

    snap = detect_snap_frame(df, id_col="stable_id")
    routes = df.copy()
    h = None
    if homography_json is not None and homography_json.exists():
        h = np.asarray(json.loads(homography_json.read_text())["H"], dtype=np.float64)
    elif source_clip is not None and source_clip.exists():
        reg = FieldRegistration()
        cap = cv2.VideoCapture(str(source_clip))
        frame_idx = 0
        while cap.isOpened() and frame_idx < 300:
            ok, frame = cap.read()
            if not ok:
                break
            reg.update(frame)
            if reg.registration_valid and reg.homography is not None:
                h = reg.homography.copy()
                break
            frame_idx += 1
        cap.release()
        if h is not None:
            auto_h = out_dir / f"{ir_path.stem}_auto_homography.json"
            auto_h.write_text(json.dumps({"H": h.tolist(), "source_clip": str(source_clip)}, indent=2))

    if h is not None:
        pts = routes.apply(lambda row: _project_foot(h, row), axis=1)
        routes["court_x"] = [p[0] for p in pts]
        routes["court_y"] = [p[1] for p in pts]

    routes = trim_routes(routes, snap_frame=snap)
    routes_path = out_dir / f"{ir_path.stem}_routes.csv"
    routes.to_csv(routes_path, index=False)

    metrics = compute_route_metrics(
        routes,
        annotations_path=annotations_csv if annotations_csv and annotations_csv.exists() else None,
    )
    metrics["snap_frame"] = int(snap)
    metrics_path = out_dir / f"{ir_path.stem}_route_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    stitched_path = out_dir / f"{ir_path.stem}_stitched.parquet"
    df.to_parquet(stitched_path, index=False)
    print(f"Saved stitched IR: {stitched_path}")
    print(f"Saved routes: {routes_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ir", required=True, help="Pass 1 parquet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--homography", default=None)
    parser.add_argument("--source-clip", default=None, help="Optional clip for auto homography fallback")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--max-gap", type=int, default=30)
    parser.add_argument("--max-dist-px", type=float, default=120.0)
    args = parser.parse_args()
    run_pass2(
        Path(args.ir),
        Path(args.out_dir),
        homography_json=Path(args.homography) if args.homography else None,
        source_clip=Path(args.source_clip) if args.source_clip else None,
        annotations_csv=Path(args.annotations) if args.annotations else None,
        max_gap=args.max_gap,
        max_dist_px=args.max_dist_px,
    )


if __name__ == "__main__":
    main()
