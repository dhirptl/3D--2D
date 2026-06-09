"""Stable team cluster identity across recalibrations (DESIGN §6.1.3)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_score


def hungarian_relabel_centroids(
    new_centroids: tuple[np.ndarray, np.ndarray],
    previous: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Match new k=2 centroids to previous team 0/1 by minimum total distance."""
    c0, c1 = new_centroids
    if previous is None:
        return c0, c1
    p0, p1 = previous
    d00 = float(np.linalg.norm(c0 - p0)) + float(np.linalg.norm(c1 - p1))
    d01 = float(np.linalg.norm(c0 - p1)) + float(np.linalg.norm(c1 - p0))
    if d01 < d00:
        return c1, c0
    return c0, c1


def cluster_silhouette(normed_features: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette for k=2 fit; None if undefined."""
    if len(normed_features) < 3 or len(set(labels)) < 2:
        return None
    try:
        return float(silhouette_score(normed_features, labels))
    except ValueError:
        return None


def lock_allowed(silhouette: float | None, min_silhouette: float) -> bool:
    """DESIGN §6.1.4: block lock when clusters are not separable."""
    if silhouette is None:
        return True
    return silhouette >= min_silhouette
