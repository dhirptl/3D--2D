"""Regression tests for clip-level offline team labels across camera cuts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.offline_teams import (
    _canonicalize_cluster_labels,
    assign_teams_offline,
)
from src.pass1_perception import run_pass1

ROOT = Path(__file__).resolve().parent.parent
SEAHAWKS_CLIP = ROOT / "la v seahawks.mp4"


def _feat(mean_h: float) -> list[float]:
    return [160.0, 100.0, 8.0, 8.0, mean_h, 5.0]


def _row(frame: int, track_id: int, mean_h: float, *, is_cut: bool = False) -> dict:
    return {
        "clip_id": "synthetic",
        "frame": frame,
        "track_id": track_id,
        "x1": 10.0,
        "y1": 10.0,
        "x2": 40.0,
        "y2": 80.0,
        "conf": 0.9,
        "mask_ref": None,
        "mask_is_fallback": False,
        "feat_lab6d": _feat(mean_h),
        "torso_px": 1000,
        "helmet_hits": 1,
        "is_cut_frame": is_cut,
        "team": -1,
        "ocr_num": None,
        "ocr_conf": 0.0,
    }


def team_flips_across_cuts(df: pd.DataFrame) -> list[tuple[int, int, int, int]]:
    """Return (track_id, cut_frame, team_before, team_after) for each flip."""
    cuts = sorted(int(v) for v in df.loc[df["is_cut_frame"] == True, "frame"].unique())
    flips: list[tuple[int, int, int, int]] = []
    for tid, g in df.groupby("track_id"):
        labeled = g[g["team"] >= 0]
        if labeled.empty:
            continue
        for cut in cuts:
            before = labeled[labeled["frame"] < cut]
            after = labeled[labeled["frame"] >= cut]
            if before.empty or after.empty:
                continue
            t0 = int(before["team"].mode().iloc[0])
            t1 = int(after["team"].mode().iloc[0])
            if t0 != t1:
                flips.append((int(tid), cut, t0, t1))
    return flips


def test_canonicalize_lower_hue_is_team_zero():
    valid = pd.DataFrame(
        {
            "conf": [0.9, 0.9],
            "torso_px": [1000, 1000],
            "feat_arr": [
                np.array(_feat(120.0)),
                np.array(_feat(25.0)),
            ],
        }
    )
    det_team = np.array([0, 1])
    out = _canonicalize_cluster_labels(valid, det_team)
    assert int(out[0]) == 1
    assert int(out[1]) == 0


def test_track_spanning_cut_keeps_one_team_label():
    rows = []
    for frame in range(80):
        is_cut = frame == 40
        for tid in range(12):
            hue = 28.0 if tid % 2 == 0 else 132.0
            if tid == 1:
                hue = 28.0
            rows.append(_row(frame, tid, hue, is_cut=is_cut))

    df = pd.DataFrame(rows)
    out = assign_teams_offline(df)
    assert team_flips_across_cuts(out) == []
    teams = out.loc[out["track_id"] == 1, "team"].unique()
    assert len(teams) == 1
    assert teams[0] >= 0


@pytest.mark.slow
def test_seahawks_no_team_flip_across_cuts(tmp_path: Path):
    if not SEAHAWKS_CLIP.exists():
        pytest.skip(f"missing clip: {SEAHAWKS_CLIP}")

    ir_path = tmp_path / "seahawks_pass1.parquet"
    run_pass1(SEAHAWKS_CLIP, ir_path, pose_every=0, helmet_model=None)
    df = pd.read_parquet(ir_path)
    out = assign_teams_offline(df, source_clip=SEAHAWKS_CLIP, ir_path=ir_path)
    flips = team_flips_across_cuts(out)
    assert flips == [], f"team label flips across cuts: {flips[:10]}"
