"""Fused association cost for BoT-SORT-style matching (DESIGN §5.2)."""

from __future__ import annotations

import numpy as np


def fuse_association_cost(
    iou: float,
    embed_dist: float,
    *,
    lambda_iou: float = 0.5,
) -> float:
    """cost = λ(1-IoU) + (1-λ)*cosine_distance."""
    lam = float(np.clip(lambda_iou, 0.0, 1.0))
    return lam * (1.0 - iou) + (1.0 - lam) * embed_dist


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (na * nb))
