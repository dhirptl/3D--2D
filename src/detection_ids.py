"""Track ID helpers without Ultralytics dependency (for unit tests)."""

from __future__ import annotations

import numpy as np

from src.config import UNTRACKED_TRACK_ID


def track_ids_from_boxes(boxes) -> np.ndarray:
    n = int(boxes.xyxy.shape[0])
    if boxes.id is None:
        return np.full(n, UNTRACKED_TRACK_ID, dtype=int)
    return boxes.id.cpu().numpy().astype(int)
