"""Temporal box interpolation on detection CSV (DESIGN §4.1.4)."""

from __future__ import annotations

import pandas as pd


def interpolate_detection_gaps(
    df: pd.DataFrame,
    *,
    max_gap: int = 5,
    id_col: str = "track_id",
    frame_col: str = "frame",
) -> pd.DataFrame:
    """Linearly interpolate bbox coordinates for short per-track gaps."""
    if df.empty:
        return df
    box_cols = ["x1", "y1", "x2", "y2"]
    for c in box_cols:
        if c not in df.columns:
            return df

    rows = []
    for tid, grp in df.groupby(id_col):
        grp = grp.sort_values(frame_col)
        frames = grp[frame_col].astype(int).tolist()
        for i in range(len(frames) - 1):
            f0, f1 = frames[i], frames[i + 1]
            gap = f1 - f0 - 1
            if gap <= 0 or gap > max_gap:
                continue
            r0 = grp[grp[frame_col] == f0].iloc[0]
            r1 = grp[grp[frame_col] == f1].iloc[0]
            for g in range(1, gap + 1):
                t = g / (gap + 1)
                row = {frame_col: f0 + g, id_col: tid}
                for c in box_cols:
                    row[c] = float(r0[c]) * (1 - t) + float(r1[c]) * t
                if "conf" in grp.columns:
                    row["conf"] = float(min(r0.get("conf", 0.5), r1.get("conf", 0.5)))
                rows.append(row)

    if not rows:
        return df
    extra = pd.DataFrame(rows)
    out = pd.concat([df, extra], ignore_index=True)
    return out.sort_values([frame_col, id_col]).reset_index(drop=True)
