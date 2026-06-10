"""Tests for degenerate homography rejection via projected polygon area."""

from __future__ import annotations

import numpy as np

from src.config import FIELD_REG_VALID_STREAK
from src.field_registration import FieldRegistration, _projection_is_plausible, _yard_match_is_degenerate

# Captured from 1minclip.mov replay (fi 191, area_frac ~1.1e-8): passes inlier/reproj
# but collapses the field template to a sliver.
DEGENERATE_H = np.array(
    [
        [-0.542554, -1.526445, 587.983333],
        [-0.243984, -0.670955, 263.769917],
        [-0.000925, -0.002544, 1.0],
    ],
    dtype=np.float64,
)

# Captured from 1minclip.mov replay (fi 189, area_frac ~0.74): proper trapezoid.
GOOD_H = np.array(
    [
        [0.952678, 0.345327, -227.460948],
        [0.300312, 0.830517, 0.718522],
        [0.000155, 0.000428, 1.0],
    ],
    dtype=np.float64,
)

FRAME_H, FRAME_W = 1080, 1920


def test_projection_rejects_degenerate_homography() -> None:
    assert _projection_is_plausible(DEGENERATE_H, FRAME_H, FRAME_W) is False


def test_projection_accepts_good_homography() -> None:
    assert _projection_is_plausible(GOOD_H, FRAME_H, FRAME_W) is True


def test_yard_match_rejects_collinear_points() -> None:
    collinear = np.array([[100.0, 500.0], [200.0, 510.0], [300.0, 520.0], [400.0, 530.0]])
    spread = np.array([[100.0, 500.0], [500.0, 520.0], [900.0, 800.0], [200.0, 900.0]])
    assert _yard_match_is_degenerate(collinear) is True
    assert _yard_match_is_degenerate(spread) is False


def test_invalidate_streak_nulls_homography() -> None:
    reg = FieldRegistration()
    reg.homography = np.eye(3, dtype=np.float64)
    reg.valid_streak = FIELD_REG_VALID_STREAK + 2
    reg.registration_valid = True
    reg._invalidate_streak()
    assert reg.homography is None
