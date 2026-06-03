"""Decomposed zero-detection metrics from ZERODET_DEBUG JSONL."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

# Pipeline bucket keys (aligned with zerodet_decompose.merge_pipeline)
ACTIONABLE_KEYS = frozenset({
    "HUD field-band ate all",
    "id=None raw>=5 (formation boundary)",
})


def load_zerodet_jsonl(path: Path) -> dict[int, dict]:
    by_frame: dict[int, dict] = {}
    if not path.exists():
        return by_frame
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_frame[int(rec["frame"])] = rec
    return by_frame


def pipeline_buckets(
    total_frames: int,
    zero_det_by_frame: dict[int, dict],
    *,
    raw_counts_by_frame: dict[int, int] | None = None,
) -> Counter:
    """Mirror zerodet_decompose.merge_pipeline without running YOLO."""
    merged: Counter = Counter()
    for frame_idx in range(total_frames):
        rec = zero_det_by_frame.get(frame_idx)
        if rec is None:
            merged["tracked/ok"] += 1
            continue
        path = rec.get("path", "OK")
        if path == "HUD_ATE_ALL":
            merged["HUD field-band ate all"] += 1
        elif path == "ID_NONE":
            rb = rec.get("raw_boxes")
            if rb is None and raw_counts_by_frame is not None:
                rb = raw_counts_by_frame.get(frame_idx, 0)
            rb = int(rb or 0)
            if rb == 0:
                merged["id=None raw=0 (empty)"] += 1
            elif rb <= 2:
                merged["id=None raw 1-2 (sparse/low-conf)"] += 1
            elif rb <= 4:
                merged["id=None raw 3-4 (boundary)"] += 1
            else:
                merged["id=None raw>=5 (formation boundary)"] += 1
        else:
            merged["tracked/ok"] += 1
    return merged


def decomposed_rates(buckets: Counter, total_frames: int) -> dict:
    total = max(total_frames, 1)
    actionable = sum(buckets.get(k, 0) for k in ACTIONABLE_KEYS)
    return {
        "pipeline_buckets": dict(buckets),
        "actionable_zero_det_frames": actionable,
        "actionable_zero_det_rate": round(actionable / total, 4),
        "hud_ate_all_frames": buckets.get("HUD field-band ate all", 0),
        "formation_boundary_frames": buckets.get("id=None raw>=5 (formation boundary)", 0),
    }


def adjusted_zero_det_rate(
    buckets: Counter,
    total_frames: int,
    *,
    exclude_raw0: bool = False,
    exclude_sparse: bool = False,
) -> float | None:
    """Headline zero-det excluding buckets treated as non-actionable montage composition."""
    if total_frames <= 0:
        return None
    zero_frames = sum(buckets.values()) - buckets.get("tracked/ok", 0)
    if exclude_raw0:
        zero_frames -= buckets.get("id=None raw=0 (empty)", 0)
    if exclude_sparse:
        zero_frames -= buckets.get("id=None raw 1-2 (sparse/low-conf)", 0)
    return round(max(zero_frames, 0) / total_frames, 4)
