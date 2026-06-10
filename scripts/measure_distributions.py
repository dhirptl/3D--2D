#!/usr/bin/env python3
"""DESIGN §11: export threshold/distribution artifacts before tuning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "measurements"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detections-csv", default=None, help="outputs/eval_detections.csv")
    ap.add_argument("--zerodet-jsonl", default=None)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"version": 1, "notes": "Fill from labeled clips; see DESIGN §11"}

    if args.detections_csv and Path(args.detections_csv).exists():
        import pandas as pd

        df = pd.read_csv(args.detections_csv)
        if "conf" in df.columns:
            conf = df["conf"].dropna().astype(float)
            report["conf_histogram"] = {
                "count": int(len(conf)),
                "p10": float(conf.quantile(0.10)),
                "p50": float(conf.quantile(0.50)),
                "p90": float(conf.quantile(0.90)),
            }
        if all(c in df.columns for c in ("x1", "y1", "x2", "y2")):
            areas = (df["x2"] - df["x1"]) * (df["y2"] - df["y1"])
            report["area_buckets"] = {
                "small_lt2k": int((areas < 2000).sum()),
                "medium": int(((areas >= 2000) & (areas < 8000)).sum()),
                "large_gte8k": int((areas >= 8000).sum()),
            }

    if args.zerodet_jsonl and Path(args.zerodet_jsonl).exists():
        from src.zerodet_metrics import (
            decomposed_rates,
            load_zerodet_jsonl,
            pipeline_buckets,
        )

        by_frame = load_zerodet_jsonl(Path(args.zerodet_jsonl))
        n_frames = max(by_frame.keys(), default=-1) + 1
        buckets = pipeline_buckets(n_frames, by_frame)
        report["zerodet_decomposed"] = decomposed_rates(buckets, n_frames)

    out_path = out_dir / "distributions.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
