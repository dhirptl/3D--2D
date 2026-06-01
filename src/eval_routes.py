"""Route-quality metrics for offline route outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def _load(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _team_accuracy(df: pd.DataFrame, annotations_path: Path) -> float | None:
    if not annotations_path.exists():
        return None
    rows = list(csv.DictReader(annotations_path.open()))
    if not rows:
        return None
    if "team" not in df.columns:
        return None
    pred = {}
    for _, row in df.iterrows():
        pred[(int(row["frame"]), int(row["track_id"]))] = int(row["team"])
    correct = 0
    total = 0
    for row in rows:
        key = (int(row["frame"]), int(row["track_id"]))
        if key not in pred:
            continue
        total += 1
        if pred[key] == int(row["correct_team"]):
            correct += 1
    if total == 0:
        return None
    return correct / total


def compute_route_metrics(df: pd.DataFrame, annotations_path: Path | None = None) -> dict:
    if df.empty:
        return {
            "tracks": 0,
            "fragmentation_rate": 0.0,
            "trajectory_completeness": 0.0,
            "id_switches": 0.0,
            "team_label_accuracy": None,
        }

    id_col = "stable_id" if "stable_id" in df.columns else "track_id"
    tracks = int(df[id_col].nunique())
    frame_span = max(1, int(df["frame"].max()) - int(df["frame"].min()) + 1)

    lengths = df.groupby(id_col)["frame"].nunique().astype(float)
    trajectory_completeness = float((lengths / frame_span).mean()) if len(lengths) else 0.0

    # Proxy fragmentation: average number of distinct IDs active per frame relative to median players/frame.
    per_frame_ids = df.groupby("frame")[id_col].nunique().astype(float)
    median_players = float(per_frame_ids.median()) if len(per_frame_ids) else 1.0
    fragmentation_rate = float(max(0.0, tracks / max(1.0, median_players) - 1.0))

    # Proxy ID switches: count discontinuities in each track's frame sequence.
    gaps = 0
    for _, g in df.groupby(id_col):
        f = np.sort(g["frame"].unique())
        if len(f) > 1:
            gaps += int(np.sum(np.diff(f) > 1))
    id_switches = float(gaps)

    team_acc = None
    if annotations_path is not None:
        team_acc = _team_accuracy(df, annotations_path)

    return {
        "tracks": tracks,
        "fragmentation_rate": fragmentation_rate,
        "trajectory_completeness": trajectory_completeness,
        "id_switches": id_switches,
        "team_label_accuracy": team_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True, help="CSV/parquet with frame, IDs, and teams")
    parser.add_argument("--annotations", default=None, help="Optional sample_labels.csv")
    parser.add_argument("--out-csv", default=None, help="Optional metric output CSV")
    args = parser.parse_args()

    df = _load(Path(args.detections))
    metrics = compute_route_metrics(
        df,
        annotations_path=Path(args.annotations) if args.annotations else None,
    )
    for k, v in metrics.items():
        print(f"{k}: {v}")

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            writer.writeheader()
            writer.writerow(metrics)
        print(f"Saved route metrics to {out}")


if __name__ == "__main__":
    main()
