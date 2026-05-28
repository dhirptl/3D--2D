"""Team classification: WARMUP -> CALIBRATING -> LOCKED (LAB 4D + k=2)."""

from collections import Counter, defaultdict, deque

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.calibration_log import CalibrationLog
from src.color_features import (
    build_raw_lab4d_vector,
    subtract_turf_from_mask,
    torso_mask_from_keypoints,
    upper_body_mask,
)
from src.config import (
    CALIBRATION_COOLDOWN_FRAMES,
    CALIBRATION_FAIL_MAX,
    CALIBRATION_MIN_NEW_SAMPLES,
    CALIBRATION_VERSION,
    CONFIDENCE_MARGIN_FRAC,
    CONFIDENCE_MARGIN_MIN,
    DEQUE_LENGTH,
    ESCAPE_MARGIN,
    ESCAPE_MARGIN_FRAC,
    ESCAPE_MARGIN_MIN,
    ESCAPE_THRESHOLD,
    EXPECTED_B_MEAN_RANGE,
    FEATURE_VERSION,
    FIELD_HSV_HUE_HIGH,
    FIELD_HSV_HUE_LOW,
    FIELD_MASK_MIN_FRAC,
    KMEANS_N_INIT,
    KMEANS_RANDOM_STATE,
    MASK_AREA_FREEZE_MULT,
    MASK_MIN_AREA_FLOOR,
    MINIMUM_CENTROID_DISTANCE,
    MINIMUM_CENTROID_DISTANCE_FALLBACK,
    MIN_UPPER_MASK_PIXELS,
    PILE_UP_DISTANCE_PX,
    PILE_UP_ESCAPE_THRESHOLD,
    POST_CUT_FRAMES,
    PREFILTER_CHROMA_MAX,
    PREFILTER_FIELD_AB_DIST,
    QUALITY_FRAMES,
    SCALER_LAB_COLS,
    SIMILAR_COLOR_AVG_DIST_THRESHOLD,
    SIMILAR_COLOR_WARN_ATTEMPTS,
    SOFT_LOCK_THRESHOLD,
    SPATIAL_CONF_MIN,
    SPATIAL_MAX_SPREAD_FRAC,
    SPATIAL_MIN_GROUP_SIZE,
    SPATIAL_MIN_PLAYERS,
    SPATIAL_MIN_REL_SEPARATION,
    TEAM_REF_AB_STD_MAX,
    TEAM_REF_MAX_MASK_AREA,
    TEAM_REF_MIN_ASPECT,
    UPPER_BODY_FRAC,
    WARMUP_CONF_HELMET,
    WARMUP_CONF_MIN,
    WARMUP_CONF_RELAXED,
    WARMUP_CONF_RELAX_AFTER_FAILS,
    WARMUP_CONFIDENCE_MARGIN,
    WARMUP_FRAME_TIMEOUT,
    WARMUP_MAX_PER_TRACK,
    WARMUP_MIN_UNIQUE_TRACKS,
    WARMUP_RELAX_AT_1200,
    WARMUP_RELAX_AT_600,
    WARMUP_SOFT_LOCK_THRESHOLD,
    WARMUP_TIMEOUT_MIN_VECTORS,
    WARMUP_VECTOR_CAP,
    FIELD_REG_TEMPLATE_H,
)
from src.field_registration import project_feet_to_field
from src.mask_utils import mask_area

_LAB_NEUTRAL = 128.0
_FIELD_HUE_REF = (FIELD_HSV_HUE_LOW + FIELD_HSV_HUE_HIGH) / 2.0


def hsv_field_to_lab_ab_centroid(
    field_hsv: tuple[int, int, int, int] | None = None,
) -> tuple[float, float]:
    """Approximate field green as LAB A/B for pre-filter distance."""
    hue = _FIELD_HUE_REF
    if field_hsv is not None:
        hue = (field_hsv[0] + field_hsv[1]) / 2.0
    # turf green in BGR -> LAB
    hsv_px = np.uint8([[[int(hue), 120, 140]]])
    bgr = cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    return float(lab[0, 0, 1]), float(lab[0, 0, 2])


def is_referee(
    raw_lab4d: np.ndarray,
    *,
    bbox: tuple[int, int, int, int] | None = None,
    mask: np.ndarray | None = None,
    locked: bool = False,
    dist_margin: float | None = None,
) -> bool:
    mean_a, mean_b = raw_lab4d[0], raw_lab4d[1]
    std_a, std_b = raw_lab4d[2], raw_lab4d[3]
    chroma = max(abs(mean_a - _LAB_NEUTRAL), abs(mean_b - _LAB_NEUTRAL))
    if chroma > PREFILTER_CHROMA_MAX:
        return False
    if (std_a + std_b) < TEAM_REF_AB_STD_MAX:
        return False
    if locked and dist_margin is not None and dist_margin >= WARMUP_CONFIDENCE_MARGIN:
        return False
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        if bh > 0 and bw / bh < TEAM_REF_MIN_ASPECT:
            return False
    if mask is not None and mask_area(mask) > TEAM_REF_MAX_MASK_AREA:
        return False
    return True


def normalize_features(
    raw_lab4d: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    cols = np.asarray(raw_lab4d, dtype=np.float64).reshape(1, -1)
    return scaler.transform(cols)[0]


def normalize_batch_features(
    vecs: np.ndarray, scaler: StandardScaler
) -> np.ndarray:
    return scaler.transform(vecs[:, SCALER_LAB_COLS])


def prefilter_calibration_vectors(
    vecs: np.ndarray,
    scores: np.ndarray,
    track_ids: list[int],
    *,
    field_ab: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Drop referees, neutral grays, and field-like samples before k=2."""
    if len(vecs) == 0:
        return vecs, scores, track_ids

    field_a, field_b = field_ab or hsv_field_to_lab_ab_centroid()
    keep = []
    for i, v in enumerate(vecs):
        mean_a, mean_b, std_a, std_b = v[0], v[1], v[2], v[3]
        chroma = max(abs(mean_a - _LAB_NEUTRAL), abs(mean_b - _LAB_NEUTRAL))
        if chroma < PREFILTER_CHROMA_MAX and (std_a + std_b) < TEAM_REF_AB_STD_MAX:
            continue
        if chroma < PREFILTER_CHROMA_MAX:
            continue
        ab_dist = ((mean_a - field_a) ** 2 + (mean_b - field_b) ** 2) ** 0.5
        if ab_dist < PREFILTER_FIELD_AB_DIST and chroma < PREFILTER_CHROMA_MAX * 2:
            continue
        keep.append(i)

    if not keep:
        return vecs, scores, track_ids

    idx = np.array(keep, dtype=int)
    return vecs[idx], scores[idx], [track_ids[i] for i in keep]


def classify_player(
    feat: np.ndarray, centroid_team0: np.ndarray, centroid_team1: np.ndarray
) -> tuple[int, float, float]:
    dist0 = float(np.linalg.norm(feat - centroid_team0))
    dist1 = float(np.linalg.norm(feat - centroid_team1))
    label = 0 if dist0 < dist1 else 1
    return label, dist0, dist1


class TemporalVoter:
    def __init__(self) -> None:
        self.histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=DEQUE_LENGTH))
        self.locked_team: dict[int, int] = {}
        self.lock_streak: dict[int, int] = defaultdict(int)
        self.esc_streak: dict[int, int] = defaultdict(int)
        self.last_label: dict[int, int] = {}
        self.mask_area_ema: dict[int, float] = {}
        self.confidence_margin = CONFIDENCE_MARGIN_MIN
        self.escape_margin = ESCAPE_MARGIN
        self.soft_lock_threshold = SOFT_LOCK_THRESHOLD

    def set_margins(self, centroid_dist: float, *, warmup: bool = False) -> None:
        if warmup:
            self.confidence_margin = WARMUP_CONFIDENCE_MARGIN
            self.escape_margin = max(ESCAPE_MARGIN_MIN, ESCAPE_MARGIN_FRAC * centroid_dist)
            self.soft_lock_threshold = WARMUP_SOFT_LOCK_THRESHOLD
        else:
            self.confidence_margin = max(
                CONFIDENCE_MARGIN_MIN, CONFIDENCE_MARGIN_FRAC * centroid_dist
            )
            self.escape_margin = max(
                ESCAPE_MARGIN_MIN, ESCAPE_MARGIN_FRAC * centroid_dist, ESCAPE_MARGIN
            )
            self.soft_lock_threshold = SOFT_LOCK_THRESHOLD

    def reset(self) -> None:
        self.histories.clear()
        self.locked_team.clear()
        self.lock_streak.clear()
        self.esc_streak.clear()
        self.last_label.clear()
        self.mask_area_ema.clear()

    def update(
        self,
        track_id: int,
        label: int,
        dist0: float,
        dist1: float,
        *,
        bbox: tuple[int, int, int, int] | None = None,
        all_bboxes: list[tuple[int, int, int, int]] | None = None,
        mask_area_val: int = 0,
    ) -> int:
        margin = abs(dist0 - dist1)
        is_confident = margin >= self.confidence_margin

        if mask_area_val > 0:
            prev = self.mask_area_ema.get(track_id, float(mask_area_val))
            self.mask_area_ema[track_id] = 0.9 * prev + 0.1 * mask_area_val

        if is_confident:
            self.histories[track_id].append(label)
            self.last_label[track_id] = label

        if track_id in self.locked_team:
            locked = self.locked_team[track_id]
            hist_avg = self.mask_area_ema.get(track_id, float(mask_area_val))
            mask_frozen = (
                mask_area_val > 0
                and hist_avg > 0
                and mask_area_val > MASK_AREA_FREEZE_MULT * hist_avg
            )
            pile_up = (
                bbox is not None
                and all_bboxes is not None
                and _is_in_pile_up(bbox, all_bboxes)
            )
            effective_escape = PILE_UP_ESCAPE_THRESHOLD if pile_up else ESCAPE_THRESHOLD

            if (
                is_confident
                and label != locked
                and margin >= self.escape_margin
                and not mask_frozen
            ):
                self.esc_streak[track_id] += 1
                self.lock_streak[track_id] = 0
                if self.esc_streak[track_id] >= effective_escape:
                    del self.locked_team[track_id]
                    self.esc_streak[track_id] = 0
                    self.lock_streak[track_id] = 0
            else:
                self.esc_streak[track_id] = 0
            return self.locked_team.get(track_id, self._majority_vote(track_id))

        if is_confident:
            hist = self.histories[track_id]
            if len(hist) >= self.soft_lock_threshold:
                recent = list(hist)[-self.soft_lock_threshold :]
                if len(set(recent)) == 1:
                    self.locked_team[track_id] = recent[0]
                    self.lock_streak[track_id] = self.soft_lock_threshold

        voted = self._majority_vote(track_id)
        if voted >= 0:
            self.last_label[track_id] = voted
        return voted

    def _majority_vote(self, track_id: int) -> int:
        hist = self.histories[track_id]
        if not hist:
            return self.last_label.get(track_id, -1)
        counts = Counter(hist)
        return counts.most_common(1)[0][0]


def _is_in_pile_up(
    bbox: tuple[int, int, int, int],
    all_bboxes: list[tuple[int, int, int, int]],
    threshold_px: float = PILE_UP_DISTANCE_PX,
) -> bool:
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    for other in all_bboxes:
        ox = (other[0] + other[2]) / 2
        oy = (other[1] + other[3]) / 2
        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
        if 0 < dist < threshold_px:
            return True
    return False


class FootballTeamClassifier:
    STATE_WARMUP = "WARMUP"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_LOCKED = "LOCKED"

    def __init__(self, *, hue_weight: float | None = None) -> None:
        del hue_weight  # legacy CLI arg; LAB features do not use W_H
        self.state = self.STATE_WARMUP
        self.warmup_vectors: list[np.ndarray] = []
        self.warmup_scores: list[float] = []
        self.warmup_track_ids: list[int] = []
        self.warmup_counts_by_track: dict[int, int] = defaultdict(int)
        self.scaler: StandardScaler | None = None
        self.centroid_team0: np.ndarray | None = None
        self.centroid_team1: np.ndarray | None = None
        self.centroid_dist: float | None = None
        self.voter = TemporalVoter()
        self.field_mask: np.ndarray | None = None
        self.homography: np.ndarray | None = None
        self.cal_log = CalibrationLog()
        self.cal_fail_count = 0
        self.centroid_threshold = MINIMUM_CENTROID_DISTANCE
        self.preview_centroids: tuple[np.ndarray, np.ndarray] | None = None
        self._calibrating = False
        self.field_hsv: tuple[int, int, int, int] | None = None
        self.field_hsv_ready = False
        self.last_frame_debug: dict[int, dict] = {}
        self.locked_frame_index: int | None = None

        self.frames_in_warmup = 0
        self.warmup_min_unique_tracks = WARMUP_MIN_UNIQUE_TRACKS
        self.calibration_cooldown = 0
        self.samples_at_last_attempt = 0
        self.calibration_attempts = 0
        self.dist_history: list[float] = []
        self.rejection_log: dict[str, int] = defaultdict(int)
        self.frame_idx = 0
        self.post_cut_frames = 0

    def load_calibration(self, data: dict) -> None:
        saved_version = data.get("feature_version", "unknown")
        if saved_version != FEATURE_VERSION:
            raise RuntimeError(
                f"Calibration version mismatch!\n"
                f"  File has version: {saved_version}\n"
                f"  Code expects: {FEATURE_VERSION}\n"
                f"  Fix: delete the .pkl file and re-run without --load-calibration"
            )
        saved_frac = data.get("upper_body_frac")
        if saved_frac is not None and saved_frac != UPPER_BODY_FRAC:
            raise RuntimeError(
                f"Config mismatch: calibration used UPPER_BODY_FRAC={saved_frac}, "
                f"current value is {UPPER_BODY_FRAC}. Regenerate calibration."
            )
        if "scaler" not in data:
            raise RuntimeError("Invalid calibration data: missing scaler")
        self.scaler = data["scaler"]
        self.centroid_team0 = data["centroid_team0"]
        self.centroid_team1 = data["centroid_team1"]
        self.centroid_dist = data.get("centroid_dist")
        if self.centroid_dist is None and self.centroid_team0 is not None:
            self.centroid_dist = float(
                np.linalg.norm(self.centroid_team0 - self.centroid_team1)
            )
        self.field_hsv = data.get("field_hsv_bounds")
        if self.field_hsv:
            self.field_hsv_ready = True
        self.state = self.STATE_LOCKED
        self.locked_frame_index = 0
        if self.centroid_dist:
            self.voter.set_margins(self.centroid_dist, warmup=False)
        self.cal_log.events.append("Loaded saved calibration")

    def calibration_dict(self) -> dict:
        return {
            "version": CALIBRATION_VERSION,
            "feature_version": FEATURE_VERSION,
            "upper_body_frac": UPPER_BODY_FRAC,
            "scaler": self.scaler,
            "centroid_team0": self.centroid_team0,
            "centroid_team1": self.centroid_team1,
            "centroid_dist": self.centroid_dist,
            "field_hsv_bounds": self.field_hsv,
        }

    def process_frame(
        self, frame_bgr: np.ndarray, detections: list[dict]
    ) -> dict[int, int]:
        results: dict[int, int] = {}
        self.last_frame_debug = {}

        if self.state == self.STATE_CALIBRATING:
            for det in detections:
                results[det["track_id"]] = -1
            return results

        all_bboxes = [tuple(d["bbox"]) for d in detections]

        if (
            self.state == self.STATE_WARMUP
            and self.homography is not None
            and detections
            and self.frame_idx % 15 == 0
        ):
            self._try_spatial_calibration(frame_bgr, detections)

        if self.state == self.STATE_WARMUP:
            self._warmup_frame_tick()

        for det in detections:
            track_id = det["track_id"]
            mask = det["mask"]
            bbox = det["bbox"]
            conf = det["conf"]

            raw_feat = build_raw_lab4d_vector(
                frame_bgr,
                mask,
                bbox,
                field_mask=self.field_mask,
                det=det,
                keypoints=det.get("keypoints"),
            )
            if raw_feat is None:
                results[track_id] = -1
                self.last_frame_debug[track_id] = {"reject": "features"}
                continue

            dist_margin = None
            if (
                self.state == self.STATE_LOCKED
                and self.scaler is not None
                and self.centroid_team0 is not None
                and self.centroid_team1 is not None
            ):
                feat = normalize_features(raw_feat, self.scaler)
                _, d0, d1 = classify_player(
                    feat, self.centroid_team0, self.centroid_team1
                )
                dist_margin = abs(d0 - d1)

            if is_referee(
                raw_feat,
                bbox=bbox,
                mask=mask,
                locked=self.state == self.STATE_LOCKED,
                dist_margin=dist_margin,
            ):
                results[track_id] = -2
                continue

            if self.state == self.STATE_WARMUP:
                reject = self._try_add_warmup_sample(track_id, det, mask, raw_feat, conf)
                if reject:
                    self.last_frame_debug[track_id] = {"reject": reject}
                self._maybe_calibrate()
                results[track_id] = self._preview_label(raw_feat, track_id)

            elif self.state == self.STATE_LOCKED:
                results[track_id] = self._classify_locked_detection(
                    track_id,
                    det,
                    raw_feat,
                    mask,
                    bbox,
                    all_bboxes,
                )

        if self.post_cut_frames > 0:
            self.post_cut_frames -= 1

        if self.state == self.STATE_WARMUP and self.frame_idx % 150 == 0 and self.rejection_log:
            total_rej = sum(self.rejection_log.values())
            print(f"[warmup] rejection breakdown (last 150 frames): total={total_rej}")
            for reason, count in self.rejection_log.items():
                print(f"  {reason}: {count}")
            self.rejection_log.clear()

        return results

    def _classify_locked_detection(
        self,
        track_id: int,
        det: dict,
        raw_feat: np.ndarray,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        all_bboxes: list[tuple[int, int, int, int]],
    ) -> int:
        assert self.scaler is not None
        assert self.centroid_team0 is not None
        assert self.centroid_team1 is not None

        feat = normalize_features(raw_feat, self.scaler)
        label, dist0, dist1 = classify_player(
            feat, self.centroid_team0, self.centroid_team1
        )
        debug_extra = {}
        if det.get("_turf_sub_skipped"):
            debug_extra["turf_sub_skipped"] = True
        if self.post_cut_frames > 0:
            result = label
            self.voter.last_label[track_id] = label
        else:
            result = self.voter.update(
                track_id,
                label,
                dist0,
                dist1,
                bbox=tuple(bbox),
                all_bboxes=all_bboxes,
                mask_area_val=mask_area(mask),
            )
        self.last_frame_debug[track_id] = {
            "dist0": dist0,
            "dist1": dist1,
            "margin": abs(dist0 - dist1),
            "raw_label": label,
            **debug_extra,
        }
        return result

    def _warmup_frame_tick(self) -> None:
        self.frames_in_warmup += 1
        if self.frames_in_warmup == WARMUP_RELAX_AT_600:
            self.warmup_min_unique_tracks = max(6, self.warmup_min_unique_tracks - 4)
            print(f"[warmup] relaxing track threshold to {self.warmup_min_unique_tracks}")
        elif self.frames_in_warmup == WARMUP_RELAX_AT_1200:
            self.warmup_min_unique_tracks = 6
            print(f"[warmup] relaxing track threshold to {self.warmup_min_unique_tracks}")
        if (
            self.frames_in_warmup > WARMUP_FRAME_TIMEOUT
            and len(self.warmup_vectors) >= WARMUP_TIMEOUT_MIN_VECTORS
        ):
            print("[warmup] TIMEOUT - forcing calibration attempt")
            self._attempt_calibrate(force=True)

    def _maybe_calibrate(self) -> None:
        if self.calibration_cooldown > 0:
            self.calibration_cooldown -= 1
            return
        if not self._ready_to_calibrate():
            return
        new_samples = len(self.warmup_vectors) - self.samples_at_last_attempt
        if new_samples < CALIBRATION_MIN_NEW_SAMPLES and self.samples_at_last_attempt > 0:
            return
        self._attempt_calibrate()

    def _attempt_calibrate(self, *, force: bool = False) -> None:
        if not force and not self._ready_to_calibrate():
            return
        self.samples_at_last_attempt = len(self.warmup_vectors)
        success = self._calibrate()
        if not success:
            self.calibration_cooldown = CALIBRATION_COOLDOWN_FRAMES

    def _effective_conf_min(self) -> float:
        if self.cal_fail_count >= WARMUP_CONF_RELAX_AFTER_FAILS:
            return WARMUP_CONF_RELAXED
        return WARMUP_CONF_MIN

    def _try_add_warmup_sample(
        self,
        track_id: int,
        det: dict,
        mask: np.ndarray,
        raw_feat: np.ndarray,
        conf: float,
    ) -> str | None:
        gate_reason = self._quality_gate_reason(det, mask)
        if gate_reason is not None:
            self.rejection_log[gate_reason] += 1
            return gate_reason
        if self.warmup_counts_by_track[track_id] >= WARMUP_MAX_PER_TRACK:
            self.rejection_log["track_cap"] += 1
            return "track_cap"

        upper_px = self._upper_mask_pixels(det, mask)
        quality_score = conf * upper_px / 500.0
        self.warmup_vectors.append(raw_feat)
        self.warmup_scores.append(quality_score)
        self.warmup_track_ids.append(track_id)
        self.warmup_counts_by_track[track_id] += 1

        if len(self.warmup_vectors) > WARMUP_VECTOR_CAP:
            worst_idx = int(np.argmin(self.warmup_scores))
            dropped_tid = self.warmup_track_ids[worst_idx]
            self.warmup_vectors.pop(worst_idx)
            self.warmup_scores.pop(worst_idx)
            self.warmup_track_ids.pop(worst_idx)
            self.warmup_counts_by_track[dropped_tid] -= 1
            if self.warmup_counts_by_track[dropped_tid] <= 0:
                del self.warmup_counts_by_track[dropped_tid]
        return None

    def _quality_gate_reason(self, det: dict, mask: np.ndarray) -> str | None:
        helmet_confirmed = det.get("helmet_confirmed", False)
        conf_min = WARMUP_CONF_HELMET if helmet_confirmed else self._effective_conf_min()
        if det["conf"] < conf_min:
            return "low_conf"
        if mask_area(mask) < MASK_MIN_AREA_FLOOR:
            return "small_mask"
        if self._upper_mask_pixels(det, mask) < MIN_UPPER_MASK_PIXELS:
            return "few_upper_pixels"
        if not helmet_confirmed and self.field_hsv_ready and self.field_mask is not None:
            if self._mask_field_overlap(det, mask) < FIELD_MASK_MIN_FRAC:
                return "not_on_field"
        return None

    def _ready_to_calibrate(self) -> bool:
        unique = len(self.warmup_counts_by_track)
        return (
            len(self.warmup_vectors) >= QUALITY_FRAMES
            and unique >= self.warmup_min_unique_tracks
        )

    def _preview_label(self, raw_feat: np.ndarray, track_id: int) -> int:
        if self.preview_centroids is None or self.scaler is None:
            return -1
        feat = normalize_features(raw_feat, self.scaler)
        label, dist0, dist1 = classify_player(
            feat, self.preview_centroids[0], self.preview_centroids[1]
        )
        dist = self.centroid_dist or float(
            np.linalg.norm(self.preview_centroids[0] - self.preview_centroids[1])
        )
        self.voter.set_margins(dist, warmup=True)
        return self.voter.update(track_id, label, dist0, dist1)

    def _frame_hw(self) -> tuple[int, int]:
        if self.field_mask is not None:
            return self.field_mask.shape[:2]
        return (0, 0)

    def _crop_mask_for_det(self, det: dict, mask: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = map(int, det["bbox"])
        fh, fw = self._frame_hw()
        if fh and fw and mask.shape[:2] == (fh, fw):
            return mask[y1:y2, x1:x2]
        return mask

    def _mask_field_overlap(self, det: dict, mask: np.ndarray) -> float:
        if self.field_mask is None:
            return 1.0
        x1, y1, x2, y2 = map(int, det["bbox"])
        fh, fw = self.field_mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(fw, x2), min(fh, y2)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        crop_mask = self._crop_mask_for_det(det, mask)
        field_roi = self.field_mask[y1:y2, x1:x2]
        if crop_mask.shape != field_roi.shape:
            return 0.0
        masked = crop_mask > 0
        n_mask = int(masked.sum())
        if n_mask == 0:
            return 0.0
        on_field = (masked & (field_roi > 0)).sum()
        return float(on_field) / n_mask

    def _upper_mask_pixels(self, det: dict, mask: np.ndarray) -> int:
        x1, y1, x2, y2 = map(int, det["bbox"])
        crop_mask = self._crop_mask_for_det(det, mask)
        turf_crop = None
        if self.field_mask is not None:
            turf_crop = self.field_mask[y1:y2, x1:x2]
        crop = det.get("frame_bgr_crop")
        player_bgr = crop if crop is not None else None
        player_mask, _ = subtract_turf_from_mask(crop_mask, turf_crop, player_bgr=player_bgr)
        torso = torso_mask_from_keypoints(
            player_mask, det.get("keypoints"), bbox=(x1, y1, x2, y2)
        )
        return mask_area(torso)

    def _reset_voter_on_lock(self) -> None:
        self.voter.reset()
        if self.centroid_dist:
            self.voter.set_margins(self.centroid_dist, warmup=False)

    def _try_spatial_calibration(
        self,
        frame_bgr: np.ndarray,
        detections: list[dict],
    ) -> bool:
        """Try locking from pre-snap spatial separation in template coordinates."""
        if self.homography is None:
            return False
        if len(detections) < SPATIAL_MIN_PLAYERS:
            return False

        labeled: list[tuple[float, np.ndarray]] = []
        for det in detections:
            if det["conf"] < SPATIAL_CONF_MIN:
                continue
            if det.get("frame_age", 0) > 0:
                continue
            pos = project_feet_to_field(tuple(det["bbox"]), self.homography)
            if pos is None:
                continue
            feat = build_raw_lab4d_vector(
                frame_bgr,
                det["mask"],
                det["bbox"],
                field_mask=self.field_mask,
                det=det,
                keypoints=det.get("keypoints"),
            )
            if feat is None:
                continue
            if is_referee(feat, bbox=tuple(det["bbox"]), mask=det["mask"]):
                continue
            labeled.append((pos[1], feat))

        if len(labeled) < SPATIAL_MIN_PLAYERS:
            return False

        field_ys = np.array([item[0] for item in labeled], dtype=np.float64).reshape(-1, 1)
        player_range = float(field_ys.max() - field_ys.min())
        if player_range <= 0.0:
            return False
        if player_range > FIELD_REG_TEMPLATE_H * SPATIAL_MAX_SPREAD_FRAC:
            return False

        km_spatial = KMeans(n_clusters=2, n_init=5, random_state=0)
        km_spatial.fit(field_ys)
        labels = km_spatial.labels_
        centers = km_spatial.cluster_centers_.flatten()
        separation = abs(float(centers[0] - centers[1]))
        relative_sep = separation / player_range

        if relative_sep < SPATIAL_MIN_REL_SEPARATION:
            print(
                f"[spatial] relative separation {relative_sep:.2f} "
                f"< {SPATIAL_MIN_REL_SEPARATION:.2f} - skipping"
            )
            return False

        group0 = [labeled[i][1] for i in range(len(labeled)) if labels[i] == 0]
        group1 = [labeled[i][1] for i in range(len(labeled)) if labels[i] == 1]
        if len(group0) < SPATIAL_MIN_GROUP_SIZE or len(group1) < SPATIAL_MIN_GROUP_SIZE:
            return False

        vecs0 = np.stack(group0)
        vecs1 = np.stack(group1)
        all_vecs = np.vstack([vecs0, vecs1])

        self.scaler = StandardScaler()
        self.scaler.fit(all_vecs[:, SCALER_LAB_COLS])

        c0 = normalize_batch_features(vecs0, self.scaler).mean(axis=0)
        c1 = normalize_batch_features(vecs1, self.scaler).mean(axis=0)
        dist = float(np.linalg.norm(c0 - c1))
        if dist <= self.centroid_threshold:
            print(
                f"[spatial] centroid dist {dist:.2f} <= "
                f"{self.centroid_threshold:.2f} - skipping"
            )
            return False

        self.centroid_team0 = c0
        self.centroid_team1 = c1
        self.preview_centroids = (c0, c1)
        self.centroid_dist = dist
        self.state = self.STATE_LOCKED
        self.locked_frame_index = self.frame_idx
        self._reset_voter_on_lock()
        print(
            f"[spatial] locked at frame {self.frame_idx} - "
            f"rel_sep {relative_sep:.2f}, dist {dist:.2f}"
        )
        self.cal_log.events.append(f"Spatial lock at frame {self.frame_idx}")
        return True

    def _calibrate(self) -> bool:
        if self._calibrating:
            return False
        self._calibrating = True
        try:
            self.state = self.STATE_CALIBRATING

            vecs = np.stack(self.warmup_vectors[-WARMUP_VECTOR_CAP:])
            scores = np.array(self.warmup_scores[-WARMUP_VECTOR_CAP:])
            tids = self.warmup_track_ids[-WARMUP_VECTOR_CAP:]

            field_ab = hsv_field_to_lab_ab_centroid(self.field_hsv)
            vecs, scores, tids = prefilter_calibration_vectors(
                vecs, scores, tids, field_ab=field_ab
            )
            if len(vecs) < QUALITY_FRAMES // 2:
                print(
                    f"[calibrate] prefilter left {len(vecs)} samples; "
                    "using unfiltered set"
                )
                vecs = np.stack(self.warmup_vectors[-WARMUP_VECTOR_CAP:])
                scores = np.array(self.warmup_scores[-WARMUP_VECTOR_CAP:])

            self.cal_log.on_calibrate_start(len(vecs))

            self.scaler = StandardScaler()
            self.scaler.fit(vecs[:, SCALER_LAB_COLS])

            b_mean = float(self.scaler.mean_[1])
            print(
                f"[scaler] LAB mean={self.scaler.mean_} scale={self.scaler.scale_}"
            )
            b_min, b_max = EXPECTED_B_MEAN_RANGE
            if not (b_min <= b_mean <= b_max):
                print(
                    f"[scaler] WARNING: unusual B mean {b_mean:.1f}. "
                    "Buffer may contain crowd/turf pixels."
                )

            normed = normalize_batch_features(vecs, self.scaler)
            # Normalize weights to sum to 1 for numerical stability
            sample_weight = scores / scores.sum() if scores.sum() > 0 else None
            km = KMeans(
                n_clusters=2,
                random_state=KMEANS_RANDOM_STATE,
                n_init=KMEANS_N_INIT,
            )
            km.fit(normed, sample_weight=sample_weight)
            c0 = km.cluster_centers_[0]
            c1 = km.cluster_centers_[1]
            dist = float(np.linalg.norm(c0 - c1))

            self.preview_centroids = (c0, c1)
            self.centroid_dist = dist
            self.voter.set_margins(dist, warmup=True)

            self.calibration_attempts += 1
            self.dist_history.append(dist)

            if self.calibration_attempts >= SIMILAR_COLOR_WARN_ATTEMPTS:
                recent = self.dist_history[-5:]
                if recent:
                    avg_dist = sum(recent) / len(recent)
                    if avg_dist < SIMILAR_COLOR_AVG_DIST_THRESHOLD:
                        print("=" * 60)
                        print("WARNING: Team colors may be too similar to separate.")
                        print(
                            f"  Average centroid distance: {avg_dist:.2f} "
                            f"(need > {self.centroid_threshold})"
                        )
                        print("=" * 60)

            if dist <= self.centroid_threshold:
                self.cal_fail_count += 1
                self.cal_log.on_calibrate_fail(
                    dist, self.centroid_threshold, self.cal_fail_count
                )
                print(self.cal_log.last_fail_reason)
                self.warmup_vectors = self.warmup_vectors[-QUALITY_FRAMES:]
                self.warmup_scores = self.warmup_scores[-QUALITY_FRAMES:]
                self.warmup_track_ids = self.warmup_track_ids[-QUALITY_FRAMES:]
                self._rebuild_track_counts()
                if self.cal_fail_count >= CALIBRATION_FAIL_MAX:
                    self.centroid_threshold = MINIMUM_CENTROID_DISTANCE_FALLBACK
                    print(f"Lowering centroid threshold to {self.centroid_threshold}")
                self.state = self.STATE_WARMUP
                return False

            self.centroid_team0 = c0
            self.centroid_team1 = c1
            self.state = self.STATE_LOCKED
            if self.locked_frame_index is None:
                self.locked_frame_index = -1
            self._reset_voter_on_lock()
            c0_n = int((km.labels_ == 0).sum())
            c1_n = int((km.labels_ == 1).sum())
            self.cal_log.on_calibrate_success(dist, c0_n, c1_n, 0)
            print(self.cal_log.events[-1])
            return True
        finally:
            self._calibrating = False

    def _rebuild_track_counts(self) -> None:
        self.warmup_counts_by_track.clear()
        for tid in self.warmup_track_ids:
            self.warmup_counts_by_track[tid] += 1

    @property
    def warmup_count(self) -> int:
        return len(self.warmup_vectors)

    @property
    def warmup_unique_tracks(self) -> int:
        return len(self.warmup_counts_by_track)
