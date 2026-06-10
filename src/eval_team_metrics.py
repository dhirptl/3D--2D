"""Team eval metrics: per-track accuracy, flips, time-to-lock (DESIGN §3.2)."""

from __future__ import annotations

from collections import defaultdict


def per_track_team_accuracy(
    annotations: list[dict],
    track_labels: dict[tuple[int, int], int],
) -> dict | None:
    """Per-track majority vote accuracy vs expected team on annotated tracks."""
    by_track: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for ann in annotations:
        tid = ann["track_id"]
        expected = ann["expected"]
        pred = track_labels.get((ann["frame"], tid), -1)
        if pred >= 0:
            by_track[tid].append((pred, expected))

    if not by_track:
        return None

    correct_tracks = 0
    total_tracks = 0
    for tid, pairs in by_track.items():
        if not pairs:
            continue
        preds = [p for p, _ in pairs]
        exp = pairs[0][1]
        majority = max(set(preds), key=preds.count)
        total_tracks += 1
        if majority == exp:
            correct_tracks += 1

    if total_tracks == 0:
        return None
    return {
        "correct_tracks": correct_tracks,
        "total_tracks": total_tracks,
        "pct": round(100 * correct_tracks / total_tracks, 1),
    }


def label_flip_count(
    frame_labels: dict[tuple[int, int], int],
    *,
    locked_only: bool = True,
) -> int:
    """Count track label changes across consecutive frames."""
    by_track: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for (fid, tid), lab in frame_labels.items():
        if lab < 0:
            continue
        by_track[tid].append((fid, lab))

    flips = 0
    for tid, series in by_track.items():
        series.sort(key=lambda x: x[0])
        for i in range(1, len(series)):
            if series[i][1] != series[i - 1][1]:
                flips += 1
    return flips
