"""Render top-down route polylines from detections + homography."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.config import FIELD_REG_TEMPLATE_H, FIELD_REG_TEMPLATE_W


def _project_foot(h: np.ndarray, row: pd.Series) -> tuple[float, float]:
    x = (float(row["x1"]) + float(row["x2"])) / 2.0
    y = float(row["y2"])
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = h @ p
    if abs(q[2]) < 1e-9:
        return float("nan"), float("nan")
    return float(q[0] / q[2]), float(q[1] / q[2])


def _load_detections(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def render_routes(
    detections_path: Path,
    homography_json: Path,
    out_path: Path,
    *,
    id_col: str = "track_id",
) -> None:
    df = _load_detections(detections_path)
    if id_col not in df.columns:
        if "stable_id" in df.columns:
            id_col = "stable_id"
        elif "track_id" in df.columns:
            id_col = "track_id"
        else:
            raise ValueError("No track_id/stable_id column found.")
    for col in ("x1", "y1", "x2", "y2", "frame"):
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    cal = json.loads(homography_json.read_text())
    h = np.asarray(cal["H"], dtype=np.float64)
    tw = int(cal.get("template_w", FIELD_REG_TEMPLATE_W))
    th = int(cal.get("template_h", FIELD_REG_TEMPLATE_H))
    canvas = np.full((th, tw, 3), 25, dtype=np.uint8)

    colors = {}
    for track_id, g in df.sort_values("frame").groupby(id_col):
        pts = []
        for _, row in g.iterrows():
            fx, fy = _project_foot(h, row)
            if np.isfinite(fx) and np.isfinite(fy):
                pts.append((int(round(fx)), int(round(fy))))
        if len(pts) < 2:
            continue
        if track_id not in colors:
            rng = np.random.default_rng(int(track_id) + 1337)
            colors[track_id] = tuple(int(v) for v in rng.integers(80, 255, size=3))
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, colors[track_id], 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="CSV/parquet with frame+bbox+track columns")
    parser.add_argument("--homography", required=True, help="Calibration JSON from calibrate_field.py")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--id-col", default="track_id", help="ID column (track_id or stable_id)")
    args = parser.parse_args()
    render_routes(
        Path(args.detections),
        Path(args.homography),
        Path(args.out),
        id_col=args.id_col,
    )
    print(f"Saved route render: {args.out}")


if __name__ == "__main__":
    main()
