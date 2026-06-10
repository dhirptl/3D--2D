"""Team cluster stability (DESIGN §6.1.3–6.1.4)."""

from __future__ import annotations

import numpy as np

from src.team_cluster_utils import (
    cluster_silhouette,
    hungarian_relabel_centroids,
    lock_allowed,
)


def test_hungarian_relabel_preserves_team_identity():
    prev = (np.array([0.0, 0.0]), np.array([10.0, 10.0]))
    # KMeans swapped cluster index
    new = (np.array([10.0, 10.0]), np.array([0.0, 0.0]))
    c0, c1 = hungarian_relabel_centroids(new, prev)
    assert np.allclose(c0, prev[0])
    assert np.allclose(c1, prev[1])


def test_lock_blocked_when_silhouette_low():
    assert lock_allowed(0.05, 0.15) is False
    assert lock_allowed(0.2, 0.15) is True
    assert lock_allowed(None, 0.15) is True


def test_cluster_silhouette_two_blobs():
    a = np.random.randn(20, 2) + np.array([0, 0])
    b = np.random.randn(20, 2) + np.array([5, 5])
    x = np.vstack([a, b])
    labels = np.array([0] * 20 + [1] * 20)
    sil = cluster_silhouette(x, labels)
    assert sil is not None
    assert sil > 0.3
