"""Post-tracker re-association: stitch new ByteTrack IDs to recently lost tracks by appearance."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np


def _enabled() -> bool:
    return os.environ.get("TRACK_REASSOC", "0").strip().lower() in ("1", "true", "yes")


def _fast_lab_ab(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Upper-torso LAB a/b mean — cheap proxy for jersey color."""
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    y_mid = y1 + max(1, int(0.35 * (y2 - y1)))
    crop = frame[y1:y_mid, x1:x2]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    return np.array([lab[:, :, 1].mean(), lab[:, :, 2].mean()], dtype=np.float64)


def _foot_xy(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float64)


@dataclass
class _LostTrack:
    canonical_id: int
    end_frame: int
    appearance: np.ndarray
    foot_xy: np.ndarray


@dataclass
class TrackReassociator:
    """Match freshly spawned tracker IDs to recently lost tracks (LAB AB + foot position)."""

    max_gap: int = 45
    max_dist_px: float = 140.0
    max_ab_dist: float = 38.0
    ab_weight: float = 0.6
    _appearance: dict[int, np.ndarray] = field(default_factory=dict)
    _last_bbox: dict[int, tuple[int, int, int, int]] = field(default_factory=dict)
    _alias: dict[int, int] = field(default_factory=dict)
    _lost: list[_LostTrack] = field(default_factory=list)
    _prev_raw_ids: set[int] = field(default_factory=set)

    def reset(self) -> None:
        self._appearance.clear()
        self._last_bbox.clear()
        self._alias.clear()
        self._lost.clear()
        self._prev_raw_ids.clear()

    def _canonical(self, tid: int) -> int:
        while tid in self._alias:
            tid = self._alias[tid]
        return tid

    def _update_appearance(self, frame: np.ndarray, det: dict) -> np.ndarray | None:
        tid = det["track_id"]
        ab = _fast_lab_ab(frame, det["bbox"])
        if ab is None:
            return self._appearance.get(tid)
        prev = self._appearance.get(tid)
        if prev is not None:
            self._appearance[tid] = 0.65 * prev + 0.35 * ab
        else:
            self._appearance[tid] = ab
        return self._appearance[tid]

    def apply(self, frame: np.ndarray, detections: list[dict], frame_idx: int) -> list[dict]:
        if not _enabled():
            return detections
        if not detections:
            self._prune_lost(frame_idx)
            return detections

        cur_raw = {d["track_id"] for d in detections}
        new_raw = cur_raw - self._prev_raw_ids
        for det in detections:
            tid = det["track_id"]
            if tid in new_raw or tid not in self._appearance or frame_idx % 12 == 0:
                self._update_appearance(frame, det)
            self._last_bbox[tid] = det["bbox"]

        for tid in self._prev_raw_ids - cur_raw:
            self._push_lost(tid, frame_idx - 1)

        self._prune_lost(frame_idx)
        used_lost: set[int] = set()

        for det in detections:
            raw_tid = det["track_id"]
            if raw_tid not in new_raw:
                continue
            app = self._appearance.get(raw_tid)
            if app is None:
                continue
            foot = _foot_xy(det["bbox"])
            best_i = None
            best_cost = float("inf")
            for i, lost in enumerate(self._lost):
                if i in used_lost:
                    continue
                gap = frame_idx - lost.end_frame
                if gap <= 0 or gap > self.max_gap:
                    continue
                dist = float(np.linalg.norm(foot - lost.foot_xy))
                if dist > self.max_dist_px:
                    continue
                ab_d = float(np.linalg.norm(app - lost.appearance))
                if ab_d > self.max_ab_dist:
                    continue
                cost = dist + self.ab_weight * ab_d
                if cost < best_cost:
                    best_cost = cost
                    best_i = i
            if best_i is not None:
                lost = self._lost[best_i]
                used_lost.add(best_i)
                canon = lost.canonical_id
                self._alias[raw_tid] = canon
                self._appearance[canon] = app
                self._last_bbox[canon] = det["bbox"]

        for det in detections:
            det["track_id"] = self._canonical(det["track_id"])

        self._prev_raw_ids = cur_raw
        return detections

    def _push_lost(self, raw_tid: int, end_frame: int) -> None:
        app = self._appearance.get(raw_tid)
        bbox = self._last_bbox.get(raw_tid)
        if app is None or bbox is None:
            return
        canon = self._canonical(raw_tid)
        self._lost.append(
            _LostTrack(
                canonical_id=canon,
                end_frame=end_frame,
                appearance=app.copy(),
                foot_xy=_foot_xy(bbox),
            )
        )

    def _prune_lost(self, frame_idx: int) -> None:
        self._lost = [
            lt for lt in self._lost if frame_idx - lt.end_frame <= self.max_gap
        ]


_default_reassoc: TrackReassociator | None = None


def get_reassociator() -> TrackReassociator:
    global _default_reassoc
    if _default_reassoc is None:
        _default_reassoc = TrackReassociator()
    return _default_reassoc
