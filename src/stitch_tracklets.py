"""Offline stitching of fragmented tracker IDs into stable IDs."""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd

from src.jersey_ocr import number_conflict

def _foot_point(row: pd.Series) -> np.ndarray:
    return np.array([(float(row["x1"]) + float(row["x2"])) / 2.0, float(row["y2"])], dtype=np.float64)


def _tail_velocity(g: pd.DataFrame) -> np.ndarray:
    if len(g) < 2:
        return np.array([0.0, 0.0], dtype=np.float64)
    g2 = g.sort_values("frame").tail(2)
    p0 = _foot_point(g2.iloc[0])
    p1 = _foot_point(g2.iloc[1])
    dt = max(1.0, float(g2.iloc[1]["frame"]) - float(g2.iloc[0]["frame"]))
    return (p1 - p0) / dt


def _feat_to_ab(v) -> np.ndarray | None:
    if v is None:
        return None
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (SyntaxError, ValueError):
            return None
    arr = np.asarray(v, dtype=np.float64)
    if arr.size < 2:
        return None
    return arr[:2]


def stitch_tracklets(
    df: pd.DataFrame,
    *,
    max_gap: int = 30,
    max_dist_px: float = 120.0,
    ab_weight: float = 0.5,
) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        out["stable_id"] = pd.Series(dtype=int)
        return out

    summaries: dict[int, dict] = {}
    for tid, g in out.groupby("track_id"):
        gs = g.sort_values("frame")
        head = gs.iloc[0]
        tail = gs.iloc[-1]
        ab_vecs = []
        for v in gs["feat_lab6d"].values:
            ab = _feat_to_ab(v)
            if ab is not None:
                ab_vecs.append(ab)
        ab_med = np.nanmedian(np.stack(ab_vecs), axis=0) if ab_vecs else np.array([np.nan, np.nan])
        team_mode = gs["team"].mode()
        team = int(team_mode.iloc[0]) if len(team_mode) else -1
        ocr_num = None
        if "ocr_num" in gs.columns:
            conf_col = gs["ocr_conf"] if "ocr_conf" in gs.columns else pd.Series(0.0, index=gs.index)
            ocr_valid = gs[(gs["ocr_num"].notna()) & (conf_col >= 0.5)]
            if len(ocr_valid):
                num_mode = ocr_valid["ocr_num"].mode()
                ocr_num = int(num_mode.iloc[0]) if len(num_mode) else None
        summaries[int(tid)] = {
            "start": int(gs["frame"].min()),
            "end": int(gs["frame"].max()),
            "head": _foot_point(head),
            "tail": _foot_point(tail),
            "vel": _tail_velocity(gs),
            "team": team,
            "ab": ab_med,
            "ocr_num": ocr_num,
        }

    ends = sorted(summaries.keys(), key=lambda t: summaries[t]["end"])
    parent: dict[int, int] = {}
    used_heads: set[int] = set()

    for a in ends:
        sa = summaries[a]
        best = None
        best_cost = float("inf")
        for b, sb in summaries.items():
            if b == a or b in used_heads:
                continue
            gap = sb["start"] - sa["end"]
            if gap <= 0 or gap > max_gap:
                continue
            if sa["team"] >= 0 and sb["team"] >= 0 and sa["team"] != sb["team"]:
                continue
            if number_conflict(sa.get("ocr_num"), sb.get("ocr_num")):
                continue
            pred = sa["tail"] + sa["vel"] * gap
            d = float(np.linalg.norm(pred - sb["head"]))
            if d > max_dist_px:
                continue
            ab_d = 0.0
            if np.isfinite(sa["ab"]).all() and np.isfinite(sb["ab"]).all():
                ab_d = float(np.linalg.norm(sa["ab"] - sb["ab"]))
            cost = d + ab_weight * ab_d
            if cost < best_cost:
                best = b
                best_cost = cost
        if best is not None:
            parent[best] = a
            used_heads.add(best)

    def root(x: int) -> int:
        while x in parent:
            x = parent[x]
        return x

    out["stable_id"] = out["track_id"].astype(int).map(root)
    return out
