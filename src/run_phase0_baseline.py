"""Operational Phase 0 baseline workflow: render routes + route metrics."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.eval_routes import compute_route_metrics
from src.render_routes import render_routes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="Detection CSV/parquet")
    parser.add_argument("--homography", required=True, help="Calibration JSON")
    parser.add_argument("--out-render", required=True, help="Top-down route PNG")
    parser.add_argument("--out-metrics", required=True, help="Route metrics CSV")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--id-col", default="track_id")
    args = parser.parse_args()

    det_path = Path(args.detections)
    render_routes(
        det_path,
        Path(args.homography),
        Path(args.out_render),
        id_col=args.id_col,
    )

    import pandas as pd
    import csv

    df = pd.read_parquet(det_path) if det_path.suffix.lower() == ".parquet" else pd.read_csv(det_path)
    metrics = compute_route_metrics(
        df,
        annotations_path=Path(args.annotations) if args.annotations else None,
    )
    out = Path(args.out_metrics)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(f"Saved render: {args.out_render}")
    print(f"Saved metrics: {args.out_metrics}")


if __name__ == "__main__":
    main()
