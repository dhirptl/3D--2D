"""GMM team clustering with BIC (DESIGN §6.2.2)."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


def fit_team_gmm(
    features: np.ndarray,
    *,
    max_components: int = 2,
) -> tuple[GaussianMixture, float]:
    """Fit 2-component GMM; return model and BIC."""
    n = len(features)
    if n < 4:
        raise ValueError("need at least 4 samples for GMM")
    gm = GaussianMixture(n_components=max_components, random_state=0)
    gm.fit(features)
    bic = float(gm.bic(features))
    return gm, bic


def gmm_team_labels(gm: GaussianMixture, features: np.ndarray) -> np.ndarray:
    return gm.predict(features)
