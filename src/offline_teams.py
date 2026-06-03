"""Offline team assignment for IR data (clip-level clustering)."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    MINIMUM_CENTROID_DISTANCE,
    OFFLINE_BRIGHTNESS_LUMINANCE_STRIDE,
    OFFLINE_BRIGHTNESS_SPLIT_DELTA,
    PREFILTER_CHROMA_MAX,
    PREFILTER_FIELD_AB_DIST,
)
from src.team_classifier import hsv_field_to_lab_ab_centroid

FEAT_HUE_IDX = 4
MIN_TEAM_SAMPLES = 8


@dataclass
class Segment:
    start: int
    end: int


@dataclass
class _KMeansFitResult:
    det_team: np.ndarray
    centers: np.ndarray
    scaler: StandardScaler


def split_on_cuts(df: pd.DataFrame) -> list[Segment]:
    """Split into contiguous frame ranges by `is_cut_frame` boundaries (lighting / cut analysis)."""
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
        if vec is None or vec.shape[0] < 5:
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


def _prepare_valid_detections(df: pd.DataFrame) -> pd.DataFrame:
    valid = df[df["mask_is_fallback"] == False].copy()
    valid["feat_arr"] = valid["feat_lab6d"].apply(_feat_to_array)
    valid = valid[valid["feat_arr"].notna()].copy()
    return _drop_referee_and_turf(valid)


def _detection_weights(valid: pd.DataFrame) -> np.ndarray:
    return (
        valid["conf"].astype(float).to_numpy()
        * np.maximum(1.0, valid["torso_px"].astype(float).to_numpy())
    )


def _fit_team_kmeans(
    valid: pd.DataFrame,
    *,
    min_centroid_dist: float,
) -> _KMeansFitResult | None:
    if len(valid) < MIN_TEAM_SAMPLES:
        return None

    x = np.stack(valid["feat_arr"].values)
    weights = _detection_weights(valid)
    scaler = StandardScaler().fit(x)
    xs = scaler.transform(x)
    km = KMeans(
        n_clusters=2,
        n_init=KMEANS_N_INIT,
        random_state=KMEANS_RANDOM_STATE,
    ).fit(xs, sample_weight=weights)

    dist = float(np.linalg.norm(km.cluster_centers_[0] - km.cluster_centers_[1]))
    if dist < min_centroid_dist:
        return None

    return _KMeansFitResult(
        det_team=km.predict(xs),
        centers=km.cluster_centers_.copy(),
        scaler=scaler,
    )


def _canonicalize_cluster_labels(valid: pd.DataFrame, det_team: np.ndarray) -> np.ndarray:
    """Team 0 = cluster with lower weighted circular mean hue (feat_lab6d index 4)."""
    out = det_team.copy()
    weights = _detection_weights(valid)
    hues = np.array([arr[FEAT_HUE_IDX] for arr in valid["feat_arr"].values], dtype=np.float64)

    mask0 = out == 0
    mask1 = out == 1
    if not mask0.any() or not mask1.any():
        return out

    h0 = float(np.average(hues[mask0], weights=weights[mask0]))
    h1 = float(np.average(hues[mask1], weights=weights[mask1]))
    if h0 > h1:
        out = 1 - out
    return out


def _canonical_centers(fit: _KMeansFitResult, valid: pd.DataFrame) -> np.ndarray:
    before = fit.det_team.copy()
    after = _canonicalize_cluster_labels(valid, before)
    centers = fit.centers.copy()
    if not np.array_equal(before, after):
        centers = centers[[1, 0]]
    return centers


def _align_labels_to_reference(
    det_team: np.ndarray,
    centers: np.ndarray,
    ref_centers: np.ndarray,
) -> np.ndarray:
    """Permute cluster ids so new fit aligns to reference canonical centers."""
    perm: list[int] = []
    for j in range(2):
        dists = [float(np.linalg.norm(ref_centers[i] - centers[j])) for i in range(2)]
        perm.append(int(np.argmin(dists)))

    if perm[0] == perm[1]:
        cost_keep = float(np.linalg.norm(ref_centers[0] - centers[0])) + float(
            np.linalg.norm(ref_centers[1] - centers[1])
        )
        cost_swap = float(np.linalg.norm(ref_centers[0] - centers[1])) + float(
            np.linalg.norm(ref_centers[1] - centers[0])
        )
        perm = [0, 1] if cost_keep <= cost_swap else [1, 0]

    mapping = {0: perm[0], 1: perm[1]}
    return np.array([mapping[int(t)] for t in det_team], dtype=det_team.dtype)


def _assign_track_teams(out: pd.DataFrame, valid: pd.DataFrame, det_team: np.ndarray) -> None:
    labeled = valid.copy()
    labeled["det_team"] = det_team
    votes = labeled.groupby("track_id")["det_team"].agg(lambda s: int(s.mode().iloc[0]))
    for tid, team in votes.items():
        out.loc[out["track_id"] == tid, "team"] = int(team)


def _segment_mean_luminance(source: Path, seg: Segment, *, stride: int) -> float:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return float("nan")

    vals: list[float] = []
    frame_idx = seg.start
    while frame_idx <= seg.end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vals.append(float(gray.mean()))
        frame_idx += max(1, stride)

    cap.release()
    return float(np.mean(vals)) if vals else float("nan")


def _brightness_split_needed(
    segments: list[Segment],
    source: Path,
    *,
    delta: float,
    stride: int,
) -> bool:
    if len(segments) <= 1:
        return False
    lums = [_segment_mean_luminance(source, seg, stride=stride) for seg in segments]
    lums = [v for v in lums if not np.isnan(v)]
    if len(lums) < 2:
        return False
    return (max(lums) - min(lums)) > delta


def _cluster_clip_level(
    out: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    min_centroid_dist: float,
) -> None:
    fit = _fit_team_kmeans(valid, min_centroid_dist=min_centroid_dist)
    if fit is None:
        return
    det_team = _canonicalize_cluster_labels(valid, fit.det_team)
    _assign_track_teams(out, valid, det_team)


def _cluster_brightness_segments(
    out: pd.DataFrame,
    segments: list[Segment],
    *,
    min_centroid_dist: float,
) -> None:
    ref_centers: np.ndarray | None = None
    for seg in segments:
        seg_df = out[(out["frame"] >= seg.start) & (out["frame"] <= seg.end)]
        valid = _prepare_valid_detections(seg_df)
        fit = _fit_team_kmeans(valid, min_centroid_dist=min_centroid_dist)
        if fit is None:
            continue

        if ref_centers is None:
            det_team = _canonicalize_cluster_labels(valid, fit.det_team)
            ref_centers = _canonical_centers(fit, valid)
        else:
            det_team = _align_labels_to_reference(fit.det_team, fit.centers, ref_centers)
            det_team = _canonicalize_cluster_labels(valid, det_team)

        _assign_track_teams(out, valid, det_team)


def resolve_source_clip(
    source_clip: Path | None,
    ir_path: Path | None = None,
) -> Path | None:
    if source_clip is not None and source_clip.exists():
        return source_clip
    if ir_path is None:
        return None
    meta_path = ir_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return None
    try:
        data = json.loads(meta_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    src = Path(str(data.get("source", "")))
    return src if src.exists() else None


def assign_teams_offline(
    df: pd.DataFrame,
    *,
    source_clip: Path | None = None,
    ir_path: Path | None = None,
    min_centroid_dist: float = MINIMUM_CENTROID_DISTANCE,
    brightness_delta: float = OFFLINE_BRIGHTNESS_SPLIT_DELTA,
    luminance_stride: int = OFFLINE_BRIGHTNESS_LUMINANCE_STRIDE,
) -> pd.DataFrame:
    """Assign teams via clip-level KMeans (+ optional brightness-gated per-segment refit)."""
    out = df.copy()
    if "team" not in out.columns:
        out["team"] = -1
    else:
        out["team"] = out["team"].fillna(-1).astype(int)

    if out.empty:
        return out

    valid_all = _prepare_valid_detections(out)
    if len(valid_all) < MIN_TEAM_SAMPLES:
        return out

    video = resolve_source_clip(source_clip, ir_path)
    segments = split_on_cuts(out)
    use_brightness_split = (
        video is not None
        and _brightness_split_needed(
            segments,
            video,
            delta=brightness_delta,
            stride=luminance_stride,
        )
    )

    if use_brightness_split:
        _cluster_brightness_segments(out, segments, min_centroid_dist=min_centroid_dist)
    else:
        _cluster_clip_level(out, valid_all, min_centroid_dist=min_centroid_dist)

    return out
