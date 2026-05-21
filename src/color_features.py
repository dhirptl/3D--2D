"""12D HSV color features from player segmentation masks (TDD Part 2)."""

import cv2
import numpy as np


def preprocess_player_crop(bgr_crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    return cv2.merge([h, s, v_eq])


def split_mask_bands(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    mid_y = (y1 + y2) // 2
    torso_mask = mask.copy()
    torso_mask[mid_y:, :] = 0
    legs_mask = mask.copy()
    legs_mask[:mid_y, :] = 0
    return torso_mask, legs_mask


def extract_band_stats(hsv_image: np.ndarray, band_mask: np.ndarray) -> np.ndarray | None:
    pixels = hsv_image[band_mask > 0]
    if len(pixels) < 10:
        return None
    return np.array([
        np.mean(pixels[:, 0]),
        np.mean(pixels[:, 1]),
        np.mean(pixels[:, 2]),
        np.std(pixels[:, 0]),
        np.std(pixels[:, 1]),
        np.std(pixels[:, 2]),
    ])


def build_raw_12d_vector(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    hsv_crop = preprocess_player_crop(crop)

    torso_mask, legs_mask = split_mask_bands(crop_mask, (0, 0, x2 - x1, y2 - y1))
    band_a = extract_band_stats(hsv_crop, torso_mask)
    band_b = extract_band_stats(hsv_crop, legs_mask)
    if band_a is None or band_b is None:
        return None
    return np.concatenate([band_a, band_b])
