"""Parity harness: compare online run_tracker output vs Pass1/Pass2 metrics."""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import pandas as pd


def _load_metrics(path: Path) -> dict:
    if path.suffix.lower() == ".json":
        import json

        return json.loads(path.read_text())
    rows = list(csv.DictReader(path.open()))
    return rows[0] if rows else {}


def run_parity(
    source: Path,
    out_dir: Path,
    *,
    model: Path | None = None,
    homography: Path | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    online_csv = out_dir / "online_detections.csv"
    offline_ir = out_dir / "pass1_ir.parquet"
    offline_metrics = out_dir / "pass1_ir_route_metrics.json"

    cmd_online = [
        "python",
        "-m",
        "src.run_tracker",
        "--source",
        str(source),
        "--out",
        str(out_dir / "online_preview.mp4"),
        "--dump-detections",
        str(online_csv),
        "--no-team",
        "--no-helmet",
    ]
    if model:
        cmd_online.extend(["--model", str(model)])
    subprocess.run(cmd_online, check=True)

    cmd_pass1 = [
        "python",
        "-m",
        "src.pass1_perception",
        "--source",
        str(source),
        "--out",
        str(offline_ir),
    ]
    if model:
        cmd_pass1.extend(["--model", str(model)])
    subprocess.run(cmd_pass1, check=True)

    cmd_pass2 = [
        "python",
        "-m",
        "src.pass2_analysis",
        "--ir",
        str(offline_ir),
        "--out-dir",
        str(out_dir),
    ]
    if homography:
        cmd_pass2.extend(["--homography", str(homography)])
    subprocess.run(cmd_pass2, check=True)

    # Compare simple detection coverage numbers.
    online = pd.read_csv(online_csv)
    online_frames = int(online["frame"].nunique()) if not online.empty else 0
    online_avg = float(len(online) / max(1, online_frames))
    off_df = pd.read_parquet(offline_ir)
    off_frames = int(off_df["frame"].nunique()) if not off_df.empty else 0
    off_avg = float(len(off_df) / max(1, off_frames))

    out = out_dir / "parity_report.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source",
                "online_frames",
                "online_avg_dets",
                "offline_frames",
                "offline_avg_rows",
                "offline_metrics_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "source": str(source),
                "online_frames": online_frames,
                "online_avg_dets": round(online_avg, 3),
                "offline_frames": off_frames,
                "offline_avg_rows": round(off_avg, 3),
                "offline_metrics_path": str(offline_metrics),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--homography", default=None)
    args = parser.parse_args()

    report = run_parity(
        Path(args.source),
        Path(args.out_dir),
        model=Path(args.model) if args.model else None,
        homography=Path(args.homography) if args.homography else None,
    )
    print(f"Saved parity report: {report}")


if __name__ == "__main__":
    main()
