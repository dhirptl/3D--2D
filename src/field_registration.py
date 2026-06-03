"""Homography-based playable-area mask from yard-line detection."""

import os

import cv2
import numpy as np

from src.config import (
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    FIELD_HSV_VAL_LOW,
    FIELD_REG_MAX_REPROJ_ERR,
    FIELD_REG_MIN_INLIERS,
    FIELD_REG_MIN_PROJ_AREA_FRAC,
    FIELD_REG_TEMPLATE_H,
    FIELD_REG_TEMPLATE_W,
    FIELD_REG_VALID_STREAK,
    FIELD_ROI_X0_FRAC,
    FIELD_ROI_X1_FRAC,
    FIELD_ROI_Y0_FRAC,
    FIELD_ROI_Y1_FRAC,
    YARD_LINE_SAT_MAX,
    YARD_LINE_VAL_MIN,
)

# NFL field: 120 yd x 53.33 yd (template pixels)
_TEMPLATE_W = FIELD_REG_TEMPLATE_W
_TEMPLATE_H = FIELD_REG_TEMPLATE_H
_HOUGH_MAX_W = 640  # downsample width before Canny+Hough for speed


def _yard_line_mask(frame: np.ndarray, hsv: np.ndarray | None = None) -> np.ndarray:
    if hsv is None:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    line_lower = np.array([0, 0, YARD_LINE_VAL_MIN])
    line_upper = np.array([179, YARD_LINE_SAT_MAX, 255])
    lines = cv2.inRange(hsv, line_lower, line_upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(lines, cv2.MORPH_CLOSE, kernel)


def _detect_line_segments(
    frame: np.ndarray, line_mask: np.ndarray
) -> list[tuple[float, float, float, float]]:
    fh, fw = frame.shape[:2]
    scale = _HOUGH_MAX_W / fw if fw > _HOUGH_MAX_W else 1.0
    if scale < 1.0:
        dw, dh = int(fw * scale), int(fh * scale)
        work_frame = cv2.resize(frame,     (dw, dh))
        work_mask  = cv2.resize(line_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
    else:
        work_frame, work_mask = frame, line_mask
    gray  = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, work_mask)
    min_len = work_frame.shape[1] // 8
    segs = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=min_len,
        maxLineGap=20,
    )
    if segs is None:
        return []
    inv = 1.0 / scale
    return [
        (x1 * inv, y1 * inv, x2 * inv, y2 * inv)
        for x1, y1, x2, y2 in segs[:, 0]
    ]


def _classify_segments(
    segments: list[tuple[float, float, float, float]],
) -> tuple[list, list]:
    if not segments:
        return [], []

    angles = [
        abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        for x1, y1, x2, y2 in segments
    ]

    # Find the dominant direction for each class by clustering around
    # the median of the roughly-horizontal and roughly-vertical subsets.
    # This lets the classifier adapt to tilted camera angles instead of
    # using fixed 25° / 65° cutoffs.
    near_horiz = [a for a in angles if a < 45 or a > 135]
    near_vert  = [a for a in angles if 45 <= a <= 135]
    h_center = float(np.median(near_horiz)) if near_horiz else 0.0
    v_center = float(np.median(near_vert))  if near_vert  else 90.0
    tol = 20.0

    horiz, vert = [], []
    for seg, angle in zip(segments, angles):
        dh = min(abs(angle - h_center), 180.0 - abs(angle - h_center))
        dv = abs(angle - v_center)
        if dh < tol:
            horiz.append(seg)
        elif dv < tol:
            vert.append(seg)
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

    horiz_sorted = sorted(horiz, key=lambda s: _line_y_at_x(s, fw / 2))
    n_h = min(len(horiz_sorted), 8)
    if n_h < 2:
        return np.empty((0, 2)), np.empty((0, 2))

    mid_x = fw / 2
    top_y    = _line_y_at_x(horiz_sorted[0],       mid_x)
    bottom_y = _line_y_at_x(horiz_sorted[n_h - 1], mid_x)
    frame_span_px = max(bottom_y - top_y, 1.0)

    # Scale the observed pixel span to the template proportionally.
    # We don't know which yard lines are visible, so we assume uniform
    # spacing and map the visible band to a matching fraction of template height.
    field_frac    = min(frame_span_px / float(fh), 0.95)
    template_span = _TEMPLATE_H * field_frac
    t_top         = (_TEMPLATE_H - template_span) / 2.0
    step_y        = template_span / (n_h - 1)

    src_pts = []
    dst_pts = []
    for i, seg in enumerate(horiz_sorted[:n_h]):
        y = _line_y_at_x(seg, mid_x)
        src_pts.append([mid_x, y])
        dst_pts.append([_TEMPLATE_W / 2, t_top + i * step_y])
        # Spread each yard line to left/right edges for stable homography
        for x_frac, t_x in ((0.15, 0.1), (0.85, 0.9)):
            x = fw * x_frac
            src_pts.append([x, _line_y_at_x(seg, x)])
            dst_pts.append([_TEMPLATE_W * t_x, t_top + i * step_y])

    if vert:
        for seg in vert[:2]:
            x_mid = (seg[0] + seg[2]) / 2
            # Map to the nearest sideline column on the template.
            # A line in the left 40% of frame → left edge (x=0), else right edge.
            template_x = 0.0 if x_mid < fw * 0.4 else float(_TEMPLATE_W)
            for y_frac in (0.35, 0.65):
                src_pts.append([x_mid, fh * y_frac])
                dst_pts.append([template_x, _TEMPLATE_H * y_frac])

    return np.array(src_pts, dtype=np.float32), np.array(dst_pts, dtype=np.float32)


def _yard_match_is_degenerate(src: np.ndarray, min_spread_ratio: float = 0.02) -> bool:
    """True when matched yard-line points are nearly collinear (unstable homography)."""
    if len(src) < FIELD_REG_MIN_INLIERS:
        return True
    centered = src - src.mean(axis=0)
    if centered.shape[0] < 3:
        return True
    try:
        _, singular, _ = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return True
    if singular[0] < 1e-6:
        return True
    return float(singular[-1] / singular[0]) < min_spread_ratio


def _field_template_polygon() -> np.ndarray:
    """Playable rectangle on template (full field including end zones)."""
    return np.array([
        [0.0,         0.0],
        [_TEMPLATE_W, 0.0],
        [_TEMPLATE_W, _TEMPLATE_H],
        [0.0,         _TEMPLATE_H],
    ], dtype=np.float32)


def _shoelace_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _projection_area_frac(H: np.ndarray, h: int, w: int) -> float | None:
    """Projected field-template area as a fraction of frame area (same path as playable_mask)."""
    if H is None or H.shape != (3, 3) or h <= 0 or w <= 0:
        return None
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    poly_t = _field_template_polygon()
    poly_f = cv2.perspectiveTransform(poly_t.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
    area = _shoelace_area(poly_f)
    if not np.isfinite(area) or area <= 0:
        return None
    frac = area / float(h * w)
    return float(frac) if np.isfinite(frac) else None


def _projection_is_plausible(H: np.ndarray, h: int, w: int) -> bool:
    frac = _projection_area_frac(H, h, w)
    return frac is not None and frac >= FIELD_REG_MIN_PROJ_AREA_FRAC


class FieldRegistration:
    """Per-video homography from broadcast frame to field template."""

    def __init__(self) -> None:
        self.homography: np.ndarray | None = None
        self.registration_valid = False
        self.valid_streak = 0
        self.homography_stale = False
        self.reproj_error: float | None = None
        self.inlier_count = 0
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self.homography = None
        self.registration_valid = False
        self.valid_streak = 0
        self.homography_stale = False
        self.reproj_error = None
        self.inlier_count = 0
        self._prev_gray = None

    def _propagate_with_camera_motion(self, frame: np.ndarray) -> None:
        """Compose lightweight frame-to-frame motion into existing homography."""
        if self.homography is None:
            self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return

        prev_pts = cv2.goodFeaturesToTrack(
            self._prev_gray,
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=8,
        )
        if prev_pts is None or len(prev_pts) < 12:
            self._prev_gray = gray
            return
        cur_pts, st, _err = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, prev_pts, None)
        if cur_pts is None or st is None:
            self._prev_gray = gray
            return
        keep = st.reshape(-1).astype(bool)
        src = prev_pts.reshape(-1, 2)[keep]
        dst = cur_pts.reshape(-1, 2)[keep]
        if len(src) < 12:
            self._prev_gray = gray
            return
        affine, inliers = cv2.estimateAffinePartial2D(
            src,
            dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )
        if affine is None or inliers is None or int(inliers.sum()) < 10:
            self._prev_gray = gray
            return
        a = np.vstack([affine, [0.0, 0.0, 1.0]])
        try:
            a_inv = np.linalg.inv(a)
            self.homography = self.homography @ a_inv
        except np.linalg.LinAlgError:
            pass
        self._prev_gray = gray

    def update(self, frame: np.ndarray, *, hsv: np.ndarray | None = None) -> bool:
        self._propagate_with_camera_motion(frame)
        fh, fw = frame.shape[:2]
        line_mask = _yard_line_mask(frame, hsv)
        segments = _detect_line_segments(frame, line_mask)
        horiz, vert = _classify_segments(segments)
        src, dst = _match_points(horiz, vert, fh, fw)

        if len(src) < FIELD_REG_MIN_INLIERS or _yard_match_is_degenerate(src):
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

        if not os.environ.get("FIELD_REG_DISABLE_PROJ_GUARD") and not _projection_is_plausible(
            H, fh, fw
        ):
            self._invalidate_streak()
            return False

        self.homography = H
        self.homography_stale = False
        self.reproj_error = mean_err
        self.inlier_count = int(inliers.sum())
        self.valid_streak += 1
        self.registration_valid = self.valid_streak >= FIELD_REG_VALID_STREAK
        return self.registration_valid

    def _invalidate_streak(self) -> None:
        self.valid_streak = max(0, self.valid_streak - 1)
        if self.valid_streak < FIELD_REG_VALID_STREAK:
            self.homography_stale = True
            self.registration_valid = False

    def playable_mask(self, h: int, w: int) -> np.ndarray | None:
        if not self.registration_valid or self.homography is None:
            return None
        if self.homography_stale:
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


def project_feet_to_field(
    bbox: tuple[int, int, int, int],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    """Project bbox bottom-center into field-template coordinates."""
    if homography.shape != (3, 3):
        return None

    x1, y1, x2, y2 = bbox
    foot_x = (float(x1) + float(x2)) / 2.0
    foot_y = float(y2)
    pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
    try:
        projected = cv2.perspectiveTransform(pt, homography)
    except cv2.error:
        return None

    fx = float(projected[0, 0, 0])
    fy = float(projected[0, 0, 1])
    if not np.isfinite(fx) or not np.isfinite(fy):
        return None
    if not (-100.0 < fx < FIELD_REG_TEMPLATE_W + 100.0):
        return None
    if not (-50.0 < fy < FIELD_REG_TEMPLATE_H + 50.0):
        return None
    return fx, fy


def center_roi_green_fraction(
    frame: np.ndarray, hsv: np.ndarray | None = None
) -> float:
    """Fraction of green pixels in center ROI (playable-area proxy for HSV deferral)."""
    h, w = frame.shape[:2]
    y0, y1 = int(h * FIELD_ROI_Y0_FRAC), int(h * FIELD_ROI_Y1_FRAC)
    x0, x1 = int(w * FIELD_ROI_X0_FRAC), int(w * FIELD_ROI_X1_FRAC)
    roi_hsv = (hsv[y0:y1, x0:x1] if hsv is not None
               else cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV))
    green = cv2.inRange(
        roi_hsv,
        np.array([FIELD_HSV_HUE_LOW,  FIELD_HSV_SAT_LOW,  FIELD_HSV_VAL_LOW]),
        np.array([FIELD_HSV_HUE_HIGH, 255,                255]),
    )
    return float(green.sum()) / (255.0 * max(green.size, 1))
