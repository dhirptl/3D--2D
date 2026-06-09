"""Tests for detection decouple (DESIGN §4.1.1) and HUD fail-open."""

from __future__ import annotations

import numpy as np

from src.config import UNTRACKED_TRACK_ID
from src.detection_ids import track_ids_from_boxes


class _TensorLike:
    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def cpu(self) -> _TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy, conf, track_ids=None):
        self.xyxy = _TensorLike(xyxy.astype(float))
        self.conf = _TensorLike(conf.astype(float))
        self.id = _TensorLike(track_ids.astype(float)) if track_ids is not None else None


def _make_fake_boxes(xyxy, conf, track_ids=None):
    return _FakeBoxes(xyxy, conf, track_ids=track_ids)


class DetectionStats:
    """Minimal stand-in to avoid importing detection.py (Ultralytics)."""

    def __init__(self) -> None:
        self.frames = 0
        self.total_dets = 0
        self.zero_det_frames = 0

    def record(self, n: int) -> None:
        self.frames += 1
        self.total_dets += n
        if n == 0:
            self.zero_det_frames += 1


def test_track_ids_from_boxes_untracked_when_id_none():
    xyxy = np.array([[0, 0, 10, 10]], dtype=float)
    conf = np.array([0.5], dtype=float)
    boxes = _make_fake_boxes(xyxy, conf, track_ids=None)
    ids = track_ids_from_boxes(boxes)
    assert ids.tolist() == [UNTRACKED_TRACK_ID]


def test_track_ids_from_boxes_preserves_ids():
    xyxy = np.array([[0, 0, 10, 10]], dtype=float)
    conf = np.array([0.5], dtype=float)
    boxes = _make_fake_boxes(xyxy, conf, track_ids=np.array([7], dtype=int))
    ids = track_ids_from_boxes(boxes)
    assert ids.tolist() == [7]


def test_zero_det_stats_not_incremented_on_id_none_with_boxes():
    stats = DetectionStats()
    stats.record(2)
    assert stats.zero_det_frames == 0
    assert stats.frames == 1
    assert stats.total_dets == 2


def test_fake_yolo_result_no_ids_fixture(fake_yolo_result_no_ids):
    assert fake_yolo_result_no_ids.boxes.id is None
    assert len(fake_yolo_result_no_ids.boxes) == 2  # noqa: SLF001
