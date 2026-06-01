"""Offline team assignment for IR data."""

from __future__ import annotations

import ast
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    MINIMUM_CENTROID_DISTANCE,
    PREFILTER_CHROMA_MAX,
    PREFILTER_FIELD_AB_DIST,
)
from src.team_classifier import hsv_field_to_lab_ab_centroid


@dataclass
class Segment:
    start: int
    end: int


def split_on_cuts(df: pd.DataFrame) -> list[Segment]:
    """Split into contiguous frame ranges by `is_cut_frame` boundaries."""
    if df.empty:
        return []
    cuts = sorted(set(int(v) for v in df.loc[df["is_cut_frame"] == True, "frame"].tolist()))
    frames = sorted(set(int(v) for v in df["frame"].tolist()))
    if not frames:
        return []
    segments: list[Segment] = []
    start = frames[0]
    last = frames[-1]
    for cut in cuts:
        if cut <= start:
            continue
        segments.append(Segment(start=start, end=cut - 1))
        start = cut
    segments.append(Segment(start=start, end=last))
    return segments


def _feat_to_array(v) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.ndarray):
        return v.astype(np.float64)
    if isinstance(v, list):
        return np.asarray(v, dtype=np.float64)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            parsed = ast.literal_eval(s)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, list):
            return np.asarray(parsed, dtype=np.float64)
        return None
    return None


def _drop_referee_and_turf(df: pd.DataFrame) -> pd.DataFrame:
    """Conservative pre-filter inspired by online classifier filters."""
    if df.empty:
        return df
    field_a, field_b = hsv_field_to_lab_ab_centroid()

    keep_idx = []
    for idx, row in df.iterrows():
        vec = _feat_to_array(row.get("feat_lab6d"))
        if vec is None or vec.shape[0] < 4:
            continue
        mean_a, mean_b, std_a, std_b = vec[:4]
        chroma = max(abs(mean_a - 128.0), abs(mean_b - 128.0))
        if chroma < PREFILTER_CHROMA_MAX:
            continue
        field_dist = float(np.hypot(mean_a - field_a, mean_b - field_b))
        if field_dist < PREFILTER_FIELD_AB_DIST:
            continue
        if (std_a + std_b) < 4.0:
            continue
        keep_idx.append(idx)
    return df.loc[keep_idx].copy()


def assign_teams_offline(
    df: pd.DataFrame,
    *,
    segments: list[Segment] | None = None,
    min_centroid_dist: float = MINIMUM_CENTROID_DISTANCE,
) -> pd.DataFrame:
    """Assign teams using pooled per-segment features and per-track majority vote."""
    out = df.copy()
    if "team" not in out.columns:
        out["team"] = -1
    else:
        out["team"] = out["team"].fillna(-1).astype(int)

    if out.empty:
        return out
    if segments is None:
        segments = split_on_cuts(out)

    for seg in segments:
        seg_df = out[(out["frame"] >= seg.start) & (out["frame"] <= seg.end)].copy()
        if seg_df.empty:
            continue
        valid = seg_df[(seg_df["mask_is_fallback"] == False)].copy()
        valid["feat_arr"] = valid["feat_lab6d"].apply(_feat_to_array)
        valid = valid[valid["feat_arr"].notna()].copy()
        valid = _drop_referee_and_turf(valid)
        if len(valid) < 8:
            continue

        x = np.stack(valid["feat_arr"].values)
        weights = (
            valid["conf"].astype(float).to_numpy()
            * np.maximum(1.0, valid["torso_px"].astype(float).to_numpy())
        )

        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        km = KMeans(
            n_clusters=2,
            n_init=KMEANS_N_INIT,
            random_state=KMEANS_RANDOM_STATE,
        ).fit(xs, sample_weight=weights)

        dist = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
        if dist < min_centroid_dist:
            continue

        det_team = km.predict(xs)
        valid = valid.assign(det_team=det_team)
        votes = valid.groupby("track_id")["det_team"].agg(lambda s: int(s.mode().iloc[0]))
        for tid, team in votes.items():
            out.loc[
                (out["frame"] >= seg.start) & (out["frame"] <= seg.end) & (out["track_id"] == tid),
                "team",
            ] = int(team)
    return out
