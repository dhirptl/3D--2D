"""HUD fail-open tests (DESIGN §4.1.3)."""

from __future__ import annotations

import numpy as np

from src.post_process import filter_hud_detections


def test_hud_filter_fail_open_when_all_rejected():
    xyxy = np.array([[0, 5, 10, 15]], dtype=float)
    confs = np.array([0.9], dtype=float)
    tids = np.array([1], dtype=int)
    kept_xyxy, _, kept_ids, _ = filter_hud_detections(
        (100, 20, 3),
        xyxy,
        confs,
        tids,
        field_mask=None,
        fail_open=True,
    )
    assert kept_xyxy.shape[0] == 1
    assert kept_ids.tolist() == [1]


def test_hud_filter_fail_closed_when_disabled():
    xyxy = np.array([[0, 5, 10, 15]], dtype=float)
    confs = np.array([0.9], dtype=float)
    tids = np.array([1], dtype=int)
    kept_xyxy, _, kept_ids, _ = filter_hud_detections(
        (100, 20, 3),
        xyxy,
        confs,
        tids,
        field_mask=None,
        fail_open=False,
    )
    assert kept_xyxy.shape[0] == 0
    assert len(kept_ids) == 0
