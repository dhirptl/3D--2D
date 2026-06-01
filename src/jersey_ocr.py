"""Lightweight jersey number OCR utilities."""

from __future__ import annotations

import re
from typing import Iterable

import cv2
import numpy as np
import pandas as pd


_DIGIT_RE = re.compile(r"\d{1,2}")


def jersey_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    bh = y2 - y1
    # Mid torso/back region
    cy1 = y1 + int(0.20 * bh)
    cy2 = y1 + int(0.75 * bh)
    if cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, x1:x2]
    return crop if crop.size else None


def ocr_number(crop: np.ndarray) -> tuple[int | None, float]:
    """Best-effort OCR. Returns (number, confidence[0,1])."""
    if crop is None or crop.size == 0:
        return None, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
    )
    # Try pytesseract if available.
    try:
        import pytesseract  # type: ignore

        text = pytesseract.image_to_string(
            thr,
            config="--psm 7 -c tessedit_char_whitelist=0123456789",
        )
        match = _DIGIT_RE.search(text)
        if not match:
            return None, 0.0
        num = int(match.group(0))
        if not (0 <= num <= 99):
            return None, 0.0
        # Conservative confidence proxy without tesseract confidences.
        return num, 0.6
    except Exception:
        return None, 0.0


def aggregate_numbers(
    df: pd.DataFrame,
    *,
    id_col: str = "stable_id",
    min_conf: float = 0.5,
) -> dict[int, int | None]:
    """Confidence-weighted per-track vote."""
    out: dict[int, int | None] = {}
    if df.empty or id_col not in df.columns:
        return out
    for sid, g in df.groupby(id_col):
        votes: dict[int, float] = {}
        for n, c in g[["ocr_num", "ocr_conf"]].dropna().itertuples(index=False):
            try:
                num = int(n)
                conf = float(c)
            except Exception:
                continue
            if conf < min_conf or not (0 <= num <= 99):
                continue
            votes[num] = votes.get(num, 0.0) + conf
        out[int(sid)] = max(votes, key=votes.get) if votes else None
    return out


def number_conflict(a_num: int | None, b_num: int | None) -> bool:
    return a_num is not None and b_num is not None and a_num != b_num
