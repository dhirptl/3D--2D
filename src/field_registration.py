"""Homography-based playable-area mask from yard-line detection."""

import cv2
import numpy as np

from src.config import (
    FIELD_REG_MAX_REPROJ_ERR,
    FIELD_REG_MIN_INLIERS,
    FIELD_REG_TEMPLATE_H,
    FIELD_REG_TEMPLATE_W,
    FIELD_REG_VALID_STREAK,
    YARD_LINE_SAT_MAX,
    YARD_LINE_VAL_MIN,
)
from src.post_process import build_field_mask

# NFL field: 120 yd x 53.33 yd (template pixels)
_TEMPLATE_W = FIELD_REG_TEMPLATE_W
_TEMPLATE_H = FIELD_REG_TEMPLATE_H


def _yard_line_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    line_lower = np.array([0, 0, YARD_LINE_VAL_MIN])
    line_upper = np.array([179, YARD_LINE_SAT_MAX, 255])
    lines = cv2.inRange(hsv, line_lower, line_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)


def _detect_line_segments(line_mask: np.ndarray) -> list[tuple[float, float, float, float]]:
    edges = cv2.Canny(line_mask, 50, 150)
    segs = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=line_mask.shape[1] // 8,
        maxLineGap=20,
    )
    if segs is None:
        return []
    out = []
    for s in segs:
        x1, y1, x2, y2 = s[0]
        out.append((float(x1), float(y1), float(x2), float(y2)))
    return out


def _classify_segments(
    segments: list[tuple[float, float, float, float]],
) -> tuple[list, list]:
    horiz, vert = [], []
    for x1, y1, x2, y2 in segments:
        dx, dy = x2 - x1, y2 - y1
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle < 25 or angle > 155:
            horiz.append((x1, y1, x2, y2))
        elif 65 < angle < 115:
            vert.append((x1, y1, x2, y2))
    return horiz, vert


def _line_y_at_x(seg: tuple[float, float, float, float], x: float) -> float:
    x1, y1, x2, y2 = seg
    if abs(x2 - x1) < 1e-3:
        return (y1 + y2) / 2
    t = (x - x1) / (x2 - x1)
    return y1 + t * (y2 - y1)


def _match_points(
    horiz: list, vert: list, fh: int, fw: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build src/dst point pairs from horizontal yard lines vs template grid."""
    if len(horiz) < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    horiz_sorted = sorted(horiz, key=lambda s: (_line_y_at_x(s, fw / 2)))
    n_h = min(len(horiz_sorted), 8)
    step_y = _TEMPLATE_H / max(n_h - 1, 1)

    src_pts = []
    dst_pts = []
    mid_x = fw / 2
    for i, seg in enumerate(horiz_sorted[:n_h]):
        y = _line_y_at_x(seg, mid_x)
        src_pts.append([mid_x, y])
        dst_pts.append([_TEMPLATE_W / 2, i * step_y])
        # Spread each yard line to left/right edges for stable homography
        for x_frac, t_x in ((0.15, 0.1), (0.85, 0.9)):
            x = fw * x_frac
            src_pts.append([x, _line_y_at_x(seg, x)])
            dst_pts.append([_TEMPLATE_W * t_x, i * step_y])

    if vert:
        for seg in vert[:2]:
            x = (seg[0] + seg[2]) / 2
            src_pts.append([x, fh * 0.5])
            dst_pts.append([_TEMPLATE_W * 0.1, _TEMPLATE_H / 2])
            src_pts.append([x, fh * 0.55])
            dst_pts.append([_TEMPLATE_W * 0.9, _TEMPLATE_H / 2])

    return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)


def _field_template_polygon() -> np.ndarray:
    """Playable rectangle on template (full field)."""
    margin_x = _TEMPLATE_W * 0.05
    margin_y = _TEMPLATE_H * 0.08
    return np.array([
        [margin_x, margin_y],
        [_TEMPLATE_W - margin_x, margin_y],
        [_TEMPLATE_W - margin_x, _TEMPLATE_H - margin_y],
        [margin_x, _TEMPLATE_H - margin_y],
    ], dtype=np.float32)


class FieldRegistration:
    """Per-video homography from broadcast frame to field template."""

    def __init__(self) -> None:
        self.homography: np.ndarray | None = None
        self.registration_valid = False
        self.valid_streak = 0
        self.reproj_error: float | None = None
        self.inlier_count = 0

    def update(self, frame: np.ndarray) -> bool:
        fh, fw = frame.shape[:2]
        line_mask = _yard_line_mask(frame)
        segments = _detect_line_segments(line_mask)
        horiz, vert = _classify_segments(segments)
        src, dst = _match_points(horiz, vert, fh, fw)

        if len(src) < FIELD_REG_MIN_INLIERS:
            self._invalidate_streak()
            return False

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            self._invalidate_streak()
            return False

        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = np.linalg.norm(proj - dst, axis=1)
        inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(src), bool)
        mean_err = float(err[inliers].mean()) if inliers.any() else float(err.mean())

        if int(inliers.sum()) < FIELD_REG_MIN_INLIERS or mean_err > FIELD_REG_MAX_REPROJ_ERR:
            self._invalidate_streak()
            return False

        self.homography = H
        self.reproj_error = mean_err
        self.inlier_count = int(inliers.sum())
        self.valid_streak += 1
        self.registration_valid = self.valid_streak >= FIELD_REG_VALID_STREAK
        return self.registration_valid

    def _invalidate_streak(self) -> None:
        self.valid_streak = max(0, self.valid_streak - 1)
        if self.valid_streak == 0:
            self.registration_valid = False

    def playable_mask(self, h: int, w: int) -> np.ndarray | None:
        if not self.registration_valid or self.homography is None:
            return None
        poly_t = _field_template_polygon()
        try:
            H_inv = np.linalg.inv(self.homography)
        except np.linalg.LinAlgError:
            return None
        poly_f = cv2.perspectiveTransform(poly_t.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, poly_f.astype(np.int32), 255)
        return mask


def center_roi_green_fraction(frame: np.ndarray) -> float:
    """Fraction of green pixels in center ROI (playable-area proxy for HSV deferral)."""
    h, w = frame.shape[:2]
    roi = frame[int(h * 0.35) : int(h * 0.85), int(w * 0.1) : int(w * 0.9)]
    mask = build_field_mask(roi)
    return float((mask > 0).sum()) / max(mask.size, 1)
