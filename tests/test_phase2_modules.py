"""Phase 2: label QA, association, GMM, tracker config."""

from __future__ import annotations

import numpy as np
import yaml

from src.association_cost import cosine_distance, fuse_association_cost
from src.config import PLAYER_PREDICT_CONF, ROOT
from src.label_qa import is_shadow_bgr
from src.team_gmm import fit_team_gmm, gmm_team_labels


def test_shadow_rejection():
    import cv2

    bgr = np.zeros((30, 30, 3), dtype=np.uint8)
    mask = np.ones((30, 30), dtype=np.uint8) * 255
    assert is_shadow_bgr(bgr, mask) is True


def test_fuse_association_cost():
    c = fuse_association_cost(0.8, 0.1, lambda_iou=0.5)
    assert 0 < c < 1


def test_cosine_distance_identical():
    a = np.array([1.0, 0.0])
    assert cosine_distance(a, a) < 1e-6


def test_gmm_bic_two_components():
    x = np.vstack([
        np.random.randn(30, 2),
        np.random.randn(30, 2) + 4,
    ])
    gm, bic = fit_team_gmm(x)
    labels = gmm_team_labels(gm, x)
    assert len(set(labels)) == 2
    assert bic < 1e6


def test_bytetrack_thresh_aligned_with_detector():
    cfg = yaml.safe_load((ROOT / "configs" / "bytetrack.yaml").read_text())
    assert cfg["new_track_thresh"] <= PLAYER_PREDICT_CONF + 0.05
    assert cfg["track_high_thresh"] <= PLAYER_PREDICT_CONF + 0.05


def test_botsort_has_reid_gmc():
    cfg = yaml.safe_load((ROOT / "configs" / "botsort.yaml").read_text())
    assert cfg.get("with_reid") is True
    assert cfg.get("gmc_method")
