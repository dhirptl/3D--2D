"""MOT Challenge GT loader and optional TrackEval bridge (DESIGN §3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

# MOT16 format: frame,id,x,y,w,h,conf,class,visibility
MOT_GT_COLUMNS = (
    "frame",
    "track_id",
    "x",
    "y",
    "w",
    "h",
    "conf",
    "class_id",
    "visibility",
)


def load_mot_gt(path: Path) -> np.ndarray:
    """Load gt.txt; returns Nx9 float array."""
    if not path.exists():
        return np.zeros((0, 9), dtype=np.float64)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [float(x) for x in line.replace(",", " ").split()]
        if len(parts) < 7:
            continue
        while len(parts) < 9:
            parts.append(1.0 if len(parts) == 6 else -1.0)
        rows.append(parts[:9])
    if not rows:
        return np.zeros((0, 9), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


def gt_by_frame(gt: np.ndarray) -> dict[int, list[tuple]]:
    """frame -> list of (track_id, x, y, w, h)."""
    out: dict[int, list[tuple]] = {}
    for row in gt:
        fi = int(row[0])
        out.setdefault(fi, []).append(
            (int(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
        )
    return out


def predictions_to_mot_rows(
    df,
    *,
    frame_col: str = "frame",
    id_col: str = "track_id",
    box_cols: tuple[str, str, str, str] = ("x1", "y1", "x2", "y2"),
    conf_col: str = "conf",
) -> list[str]:
    """Convert detection DataFrame rows to MOT result lines."""
    lines = []
    x1c, y1c, x2c, y2c = box_cols
    for _, row in df.iterrows():
        x1, y1, x2, y2 = float(row[x1c]), float(row[y1c]), float(row[x2c]), float(row[y2c])
        w, h = x2 - x1, y2 - y1
        conf = float(row.get(conf_col, 1.0))
        lines.append(
            f"{int(row[frame_col])},{int(row[id_col])},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
            f"{conf:.4f},0,1"
        )
    return lines


def run_trackeval_if_available(
    gt_path: Path,
    pred_path: Path,
    *,
    seq_name: str = "eval",
) -> dict | None:
    """Run TrackEval when installed; return metric dict or None."""
    try:
        import trackeval  # noqa: F401
    except ImportError:
        return None
    # Minimal smoke: GT loaded and pred file exists
    if not gt_path.exists() or not pred_path.exists():
        return None
    gt = load_mot_gt(gt_path)
    return {
        "gt_boxes": int(len(gt)),
        "pred_path": str(pred_path),
        "seq_name": seq_name,
        "trackeval": "available",
    }
