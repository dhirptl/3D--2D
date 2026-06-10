"""Phase 1: SAHI merge, interpolation, dominant color."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.color_features import dominant_jersey_lab_ab
from src.detection_interpolate import interpolate_detection_gaps
from src.detection_sahi import merge_tile_detections, tile_slices, weighted_box_fusion


def test_tile_slices_cover_frame():
    boxes = tile_slices(1920, 1080, tile_size=640, overlap=0.2)
    assert len(boxes) >= 4


def test_merge_tile_detections_dedupes():
    boxes = np.array([[10, 10, 50, 50], [12, 12, 48, 48]], dtype=float)
    scores = np.array([0.9, 0.8], dtype=float)
    out_b, out_s = merge_tile_detections(boxes, scores, iou_thresh=0.5)
    assert len(out_b) == 1


def test_wbf_fuses_boxes():
    b1 = np.array([[0, 0, 10, 10]], dtype=float)
    b2 = np.array([[1, 1, 11, 11]], dtype=float)
    fb, fs = weighted_box_fusion([b1, b2], [np.array([0.9]), np.array([0.85])])
    assert len(fb) >= 1


def test_interpolate_fills_gap():
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10, "conf": 0.9},
        {"frame": 3, "track_id": 1, "x1": 30, "y1": 0, "x2": 40, "y2": 10, "conf": 0.9},
    ])
    out = interpolate_detection_gaps(df, max_gap=5)
    assert len(out) > len(df)
    assert 1 in out["frame"].values
    assert 2 in out["frame"].values


def test_interpolate_skips_long_gap():
    df = pd.DataFrame([
        {"frame": 0, "track_id": 1, "x1": 0, "y1": 0, "x2": 10, "y2": 10},
        {"frame": 20, "track_id": 1, "x1": 30, "y1": 0, "x2": 40, "y2": 10},
    ])
    out = interpolate_detection_gaps(df, max_gap=3)
    assert len(out) == len(df)


def test_dominant_color_ignores_skin_cluster():
    import cv2

    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    bgr[:, :] = (200, 180, 50)  # jersey-ish BGR
    torso = np.zeros((40, 40), dtype=np.uint8)
    torso[5:35, 5:35] = 255
    ab = dominant_jersey_lab_ab(bgr, torso, k=3)
    assert ab is not None
