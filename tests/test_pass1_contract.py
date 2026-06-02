"""Contract tests for Pass1 IR schema and quality gate sanity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.pass1_quality_gate import evaluate_ir


def _base_row() -> dict:
    return {
        "clip_id": "clip",
        "frame": 0,
        "track_id": 1,
        "x1": 10.0,
        "y1": 10.0,
        "x2": 40.0,
        "y2": 80.0,
        "conf": 0.9,
        "mask_ref": None,
        "mask_is_fallback": False,
        "feat_lab6d": [1, 2, 3, 4, 5, 6],
        "torso_px": 1000,
        "helmet_hits": 1,
        "is_cut_frame": False,
        "team": -1,
        "ocr_num": None,
        "ocr_conf": 0.0,
    }


def test_pass1_contract_missing_columns(tmp_path: Path):
    df = pd.DataFrame([{"clip_id": "clip", "frame": 0}])
    path = tmp_path / "ir.parquet"
    df.to_parquet(path, index=False)
    out = evaluate_ir(path)
    assert out["pass"] is False
    assert out["reason"] == "missing_columns"


def test_pass1_contract_value_sanity(tmp_path: Path):
    rows = []
    for frame in range(12):
        for tid in range(12):
            row = _base_row()
            row["frame"] = frame
            row["track_id"] = tid
            row["x1"] = 10.0 + tid
            row["x2"] = 40.0 + tid
            rows.append(row)
    path = tmp_path / "ir.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    out = evaluate_ir(path)
    assert out["pass"] is True
    assert out["metrics"]["median_det_per_frame"] >= 12.0
    assert out["metrics"]["unique_tracks"] >= 8
