"""Snap detection and route anchoring utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_snap_frame(
    df: pd.DataFrame,
    *,
    id_col: str = "stable_id",
    quiet_win: int = 10,
    jump_factor: float = 3.0,
) -> int:
    if df.empty:
        return 0
    work = df.copy()
    if id_col not in work.columns:
        id_col = "track_id"

    work["cx"] = (work["x1"].astype(float) + work["x2"].astype(float)) / 2.0
    work["cy"] = work["y2"].astype(float)
    work = work.sort_values([id_col, "frame"])
    dx = work.groupby(id_col)["cx"].diff()
    dy = work.groupby(id_col)["cy"].diff()
    work["speed"] = np.sqrt(dx.pow(2) + dy.pow(2))
    per_frame = work.groupby("frame")["speed"].median().fillna(0.0).sort_index()
    if len(per_frame) <= quiet_win:
        return int(per_frame.index.min())
    baseline = per_frame.rolling(quiet_win).median().fillna(0.0)
    for frame in per_frame.index[quiet_win:]:
        thresh = jump_factor * max(float(baseline.loc[frame]), 1e-3)
        if float(per_frame.loc[frame]) > thresh:
            return int(frame)
    return int(per_frame.index.min())


def trim_routes(
    routes: pd.DataFrame,
    *,
    snap_frame: int,
    play_len: int = 240,
) -> pd.DataFrame:
    if routes.empty:
        return routes.copy()
    out = routes[(routes["frame"] >= snap_frame) & (routes["frame"] <= snap_frame + play_len)].copy()
    out["t_from_snap"] = out["frame"].astype(int) - int(snap_frame)
    return out
