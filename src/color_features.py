"""LAB chrominance features from segmentation masks (isolate, turf subtract, torso, L CLAHE)."""

import cv2
import numpy as np

from src.config import (
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_HSV_SAT_LOW,
    GREEN_SAFEGUARD_AB_STD_MAX,
    GREEN_SAFEGUARD_HSV_FRAC,
    MASK_ERODE_PX,
    MIN_COLOR_PIXELS,
    MIN_UPPER_MASK_PIXELS,
    UPPER_BODY_FRAC,
)
from src.mask_utils import erode_mask, mask_area

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# OpenCV LAB neutral A/B
_LAB_NEUTRAL = 128.0


def subtract_turf_from_mask(
    crop_mask: np.ndarray,
    turf_mask_crop: np.ndarray | None,
    *,
    player_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """
    Remove turf/yard-line pixels from player mask.
    Returns (mask, turf_sub_applied). Skips subtract when green-safeguard triggers.
    """
    if turf_mask_crop is None or not np.any(turf_mask_crop):
        return crop_mask, False
    if turf_mask_crop.shape != crop_mask.shape:
        return crop_mask, False

    if player_bgr is not None and _should_bypass_turf_subtract(player_bgr, crop_mask):
        return crop_mask, False

    out = cv2.bitwise_and(crop_mask, cv2.bitwise_not(turf_mask_crop))
    # Guard: if the field mask (inflated by 51x51 morph close) wipes out >75% of the
    # player seg mask, skip subtraction — the close filled in the player body as "turf".
    if mask_area(out) < mask_area(crop_mask) * 0.25:
        return crop_mask, False
    return out, True


def _should_bypass_turf_subtract(player_bgr: np.ndarray, crop_mask: np.ndarray) -> bool:
    """Protect green jerseys (Eagles/Packers) from turf erasure."""
    masked = crop_mask > 0
    if masked.sum() < MIN_COLOR_PIXELS:
        return False
    hsv = cv2.cvtColor(player_bgr, cv2.COLOR_BGR2HSV)
    px = hsv[masked]
    green = (
        (px[:, 0] >= FIELD_HSV_HUE_LOW)
        & (px[:, 0] <= FIELD_HSV_HUE_HIGH)
        & (px[:, 1] >= FIELD_HSV_SAT_LOW)
    )
    green_frac = float(green.sum()) / len(px)
    if green_frac < GREEN_SAFEGUARD_HSV_FRAC:
        return False
    lab = cv2.cvtColor(player_bgr, cv2.COLOR_BGR2LAB)
    ab = lab[masked][:, 1:3].astype(np.float32)
    if ab.shape[0] < MIN_COLOR_PIXELS:
        return False
    return float(ab[:, 0].std() + ab[:, 1].std()) < GREEN_SAFEGUARD_AB_STD_MAX


def upper_body_mask(
    crop_mask: np.ndarray,
    top_frac: float = UPPER_BODY_FRAC,
    *,
    bbox: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """Keep upper-body region; adapt crop for horizontal (diving) players."""
    ys = np.where(crop_mask > 0)[0]
    if len(ys) == 0:
        return crop_mask.copy()
    y_min, y_max = int(ys.min()), int(ys.max())
    span = y_max - y_min + 1

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        box_h = max(y2 - y1, 1)
        box_w = x2 - x1
        aspect = box_w / box_h
        if aspect > 1.3:
            crop_start = int(span * 0.2)
            crop_end = int(span * 0.8)
        elif aspect < 0.5:
            crop_start = 0
            crop_end = int(span * top_frac)
        else:
            t = min(1.0, max(0.0, (aspect - 0.5) / 0.8))
            top_end = int(span * top_frac)
            mid_end = int(span * 0.8)
            crop_start = 0
            crop_end = int(top_end * (1 - t) + mid_end * t)
    else:
        crop_start = 0
        crop_end = int(span * top_frac)

    out = crop_mask.copy()
    out[: y_min + crop_start, :] = 0
    out[y_min + crop_end :, :] = 0
    return out


def torso_mask_from_keypoints(
    crop_mask: np.ndarray,
    keypoints: np.ndarray | None,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    min_kpt_conf: float = 0.3,
) -> np.ndarray:
    """
    Torso polygon from COCO shoulders (5,6) and hips (11,12), intersected with seg mask.
    Falls back to aspect-aware upper_body_mask when keypoints are missing.
    """
    if keypoints is None or len(keypoints) < 17:
        return upper_body_mask(crop_mask, bbox=bbox)

    # keypoints: (17, 3) x, y, conf
    idx = [5, 6, 12, 11]
    pts = []
    for i in idx:
        x, y, c = keypoints[i]
        if c < min_kpt_conf:
            return upper_body_mask(crop_mask, bbox=bbox)
        pts.append([int(x), int(y)])

    h, w = crop_mask.shape[:2]
    poly = np.array(pts, dtype=np.int32)
    poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
    torso = np.zeros_like(crop_mask)
    cv2.fillConvexPoly(torso, poly, 255)
    return cv2.bitwise_and(crop_mask, torso)


def preprocess_lab_clahe(player_bgr: np.ndarray) -> np.ndarray:
    """Equalize LAB L channel only; A/B chrominance unchanged."""
    lab = cv2.cvtColor(player_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_eq = _CLAHE.apply(l_ch)
    return cv2.merge([l_eq, a_ch, b_ch])


def extract_lab_ab_stats(lab_image: np.ndarray, band_mask: np.ndarray) -> np.ndarray | None:
    """4D feature: mean_A, mean_B, std_A, std_B."""
    pixels = lab_image[band_mask > 0]
    if len(pixels) < MIN_COLOR_PIXELS:
        return None
    a = pixels[:, 1].astype(np.float32)
    b = pixels[:, 2].astype(np.float32)
    return np.array([a.mean(), b.mean(), a.std(), b.std()], dtype=np.float64)


def build_raw_lab4d_vector(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    field_mask: np.ndarray | None = None,
    det: dict | None = None,
    keypoints: np.ndarray | None = None,
) -> np.ndarray | None:
    """
    Color pipeline: isolate -> turf subtract (green safeguard) -> torso -> LAB L CLAHE -> 4D AB.
    """
    if det is not None and det.get("frame_age", 0) > 0:
        return None

    x1, y1, x2, y2 = map(int, bbox)
    fh, fw = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    crop_h, crop_w = y2 - y1, x2 - x1
    if mask.shape[:2] == (fh, fw):
        crop_mask = mask[y1:y2, x1:x2]
    else:
        crop_mask = mask.copy()

    if crop_mask.shape[:2] != (crop_h, crop_w):
        return None

    if MASK_ERODE_PX > 0:
        crop_mask = erode_mask(crop_mask, MASK_ERODE_PX)

    player_bgr = cv2.bitwise_and(crop, crop, mask=crop_mask)

    turf_crop = None
    if field_mask is not None:
        turf_crop = field_mask[y1:y2, x1:x2]
    player_mask, turf_skipped = subtract_turf_from_mask(
        crop_mask, turf_crop, player_bgr=player_bgr
    )

    kpts = keypoints
    if det is not None and kpts is None:
        kpts = det.get("keypoints")
    torso = torso_mask_from_keypoints(
        player_mask, kpts, bbox=(x1, y1, x2, y2)
    )
    if mask_area(torso) < MIN_UPPER_MASK_PIXELS:
        return None

    lab_crop = preprocess_lab_clahe(player_bgr)
    vec = extract_lab_ab_stats(lab_crop, torso)
    if vec is not None and det is not None:
        det["_turf_sub_skipped"] = turf_skipped
    return vec


# Backward-compatible alias
build_raw_6d_vector = build_raw_lab4d_vector

# Legacy HSV path for tests that still reference it
def preprocess_hsv_v_clahe(player_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(player_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_enhanced = _CLAHE.apply(v_ch)
    return cv2.merge([h_ch, s_ch, v_enhanced])
