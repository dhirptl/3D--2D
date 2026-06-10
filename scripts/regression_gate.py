#!/usr/bin/env python3
"""DESIGN §3.4: fail if clip metrics regress vs tests/baselines/default.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.regression_gate import compare_clip, load_baselines  # noqa: E402

BASELINES_PATH = ROOT / "tests" / "baselines" / "default.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="Pairs clip_id:path/to/report.json",
    )
    ap.add_argument("--baselines", default=str(BASELINES_PATH))
    ap.add_argument("--tolerance-pct", type=float, default=None)
    args = ap.parse_args()

    baselines = load_baselines(Path(args.baselines))
    tol = args.tolerance_pct if args.tolerance_pct is not None else float(
        baselines.get("tolerance_pct", 5.0)
    )

    all_errors: list[str] = []
    for spec in args.reports:
        if ":" not in spec:
            print(f"Bad --reports entry (need clip_id:path): {spec}", file=sys.stderr)
            sys.exit(2)
        clip_id, path = spec.split(":", 1)
        report = json.loads(Path(path).read_text())
        all_errors.extend(compare_clip(clip_id, report, baselines, tol))

    if all_errors:
        for e in all_errors:
            print(f"REGRESSION: {e}", file=sys.stderr)
        sys.exit(1)
    print("regression_gate: OK")


if __name__ == "__main__":
    main()
