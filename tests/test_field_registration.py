"""Tests for degenerate homography rejection via projected polygon area."""

from __future__ import annotations

import numpy as np

from src.field_registration import _projection_is_plausible

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
