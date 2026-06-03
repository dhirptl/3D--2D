"""Tests for decomposed zero-det metrics."""

from __future__ import annotations

from src.zerodet_metrics import decomposed_rates, pipeline_buckets


def test_actionable_zero_det_counts_hud_and_formation() -> None:
    by_frame = {
        1: {"path": "HUD_ATE_ALL", "raw_boxes": 2},
        2: {"path": "ID_NONE", "raw_boxes": 0},
        3: {"path": "ID_NONE", "raw_boxes": 6},
    }
    buckets = pipeline_buckets(4, by_frame)
    rates = decomposed_rates(buckets, 4)
    assert rates["actionable_zero_det_frames"] == 2
    assert rates["hud_ate_all_frames"] == 1
    assert rates["formation_boundary_frames"] == 1
