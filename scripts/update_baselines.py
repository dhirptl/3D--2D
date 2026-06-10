#!/usr/bin/env python3
"""Write regression baselines from eval_clip reports (requires --confirm)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_report_schema import extract_baseline_metrics  # noqa: E402

BASELINES_PATH = ROOT / "tests" / "baselines" / "default.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True, help="Baseline key (e.g. arz_nyj_1min)")
    ap.add_argument("--report-json", required=True, help="eval_clip report JSON path")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="Required to overwrite baselines (destructive)",
    )
    args = ap.parse_args()
    if not args.confirm:
        print("Refusing to update baselines without --confirm", file=sys.stderr)
        sys.exit(2)

    report = json.loads(Path(args.report_json).read_text())
    metrics = extract_baseline_metrics(report)

    data = json.loads(BASELINES_PATH.read_text()) if BASELINES_PATH.exists() else {
        "version": 1,
        "tolerance_pct": 5.0,
        "clips": {},
    }
    data["clips"][args.clip_id] = metrics
    BASELINES_PATH.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Updated baseline for {args.clip_id} -> {BASELINES_PATH}")


if __name__ == "__main__":
    main()
