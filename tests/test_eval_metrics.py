"""Eval harness schema and decomposition (DESIGN §3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval_mot import load_mot_gt
from src.eval_report_schema import (
    BASELINE_METRIC_KEYS,
    extract_baseline_metrics,
    validate_report_schema,
)
from src.eval_team_metrics import label_flip_count, per_track_team_accuracy


def test_eval_report_schema_minimal():
    report = {
        "frames": 100,
        "avg_detections": 12.0,
        "zero_det_rate": 0.1,
        "track_id_change_rate": 0.3,
        "label_flip_rate": 0.05,
        "locked_pct": 90.0,
        "model_miss_rate": 0.02,
        "id_none_dropped_rate": 0.01,
        "hud_dropped_rate": 0.03,
    }
    assert validate_report_schema(report) == []


def test_mot_gt_loader_missing():
    gt = load_mot_gt(Path("/nonexistent/gt.txt"))
    assert gt.shape == (0, 9)


def test_mot_gt_loader_sample(tmp_path: Path):
    p = tmp_path / "gt.txt"
    p.write_text("1,3,10,20,30,40,1,-1,1\n")
    gt = load_mot_gt(p)
    assert len(gt) == 1
    assert int(gt[0, 1]) == 3


def test_per_track_team_accuracy():
    anns = [{"frame": 1, "track_id": 1, "expected": 0}]
    labels = {(1, 1): 0, (2, 1): 0}
    acc = per_track_team_accuracy(anns, labels)
    assert acc is not None
    assert acc["pct"] == 100.0


def test_trackeval_adapter_smoke():
    from src.eval_mot import run_trackeval_if_available

    assert run_trackeval_if_available(Path("x"), Path("y")) is None


def test_extract_baseline_metrics():
    report = {"avg_detections": 3.0, "zero_det_rate": 0.1, "extra": 1}
    m = extract_baseline_metrics(report)
    assert "avg_detections" in m
    assert "extra" not in m
    assert set(m.keys()).issubset(set(BASELINE_METRIC_KEYS))
