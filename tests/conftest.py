"""Shared pytest fixtures for DESIGN TDD."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FIXTURES = Path(__file__).resolve().parent / "fixtures"
BASELINES = Path(__file__).resolve().parent / "baselines"


@dataclass
class FakeBoxes:
    xyxy: Any
    conf: Any
    id: Any | None

    def __len__(self) -> int:
        return len(self.xyxy)


@dataclass
class FakeMasks:
    data: Any

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


@dataclass
class FakeYOLOResult:
    boxes: FakeBoxes | None
    masks: FakeMasks | None = None


class _TensorLike:
    """Minimal CPU tensor stand-in for Ultralytics boxes."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.asarray(arr)

    def cpu(self) -> _TensorLike:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr

    def __len__(self) -> int:
        return len(self._arr)


def make_fake_boxes(
    xyxy: np.ndarray,
    conf: np.ndarray,
    *,
    track_ids: np.ndarray | None = None,
) -> FakeBoxes:
    n = len(xyxy)
    if track_ids is None:
        tid = None
    else:
        tid = _TensorLike(track_ids.astype(float))
    return FakeBoxes(
        xyxy=_TensorLike(xyxy.astype(float)),
        conf=_TensorLike(conf.astype(float)),
        id=tid,
    )


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_boxes() -> tuple[np.ndarray, np.ndarray]:
    xyxy = np.array([[10.0, 40.0, 30.0, 80.0], [50.0, 45.0, 70.0, 85.0]], dtype=float)
    conf = np.array([0.9, 0.85], dtype=float)
    return xyxy, conf


@pytest.fixture
def fake_yolo_result_no_ids(synthetic_boxes) -> FakeYOLOResult:
    xyxy, conf = synthetic_boxes
    return FakeYOLOResult(boxes=make_fake_boxes(xyxy, conf, track_ids=None))


@pytest.fixture
def fake_yolo_result_with_ids(synthetic_boxes) -> FakeYOLOResult:
    xyxy, conf = synthetic_boxes
    return FakeYOLOResult(
        boxes=make_fake_boxes(xyxy, conf, track_ids=np.array([1, 2], dtype=int))
    )


@pytest.fixture
def zerodet_jsonl_fixture() -> dict[int, dict]:
    path = FIXTURES / "zerodet" / "sample.jsonl"
    by_frame: dict[int, dict] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            by_frame[int(rec["frame"])] = rec
    return by_frame


@pytest.fixture
def baseline_dir() -> Path:
    BASELINES.mkdir(parents=True, exist_ok=True)
    return BASELINES
