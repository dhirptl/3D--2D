"""Validate feature-version migration compatibility for calibration artifacts."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from src.config import FEATURE_VERSION


def check_calibration(path: Path) -> dict:
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:
        return {"path": str(path), "ok": False, "error": str(e)}
    version = data.get("feature_version")
    ok = version == FEATURE_VERSION
    return {
        "path": str(path),
        "ok": ok,
        "artifact_feature_version": version,
        "expected_feature_version": FEATURE_VERSION,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="Calibration pickle paths")
    args = parser.parse_args()
    all_ok = True
    for p in args.paths:
        r = check_calibration(Path(p))
        print(r)
        all_ok &= bool(r["ok"])
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
