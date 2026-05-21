"""Team classification: WARMUP -> CALIBRATING -> LOCKED (TDD Parts 1, 3-7)."""

from collections import defaultdict, deque

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.calibration_log import CalibrationLog
from src.color_features import build_raw_12d_vector
from src.config import (
    CALIBRATION_FAIL_MAX,
    CONFIDENCE_MARGIN,
    DEQUE_LENGTH,
    ESCAPE_MARGIN,
    ESCAPE_THRESHOLD,
    MINIMUM_CENTROID_DISTANCE,
    MINIMUM_CENTROID_DISTANCE_FALLBACK,
    QUALITY_FRAMES,
    SCALER_SV_COLS,
    SOFT_LOCK_THRESHOLD,
    TEAM_REF_S_MAX,
    TEAM_REF_STD_V_MIN,
    W_B,
    W_H,
    WARMUP_CONF_MIN,
)


def is_referee(raw_12d: np.ndarray) -> bool:
    torso_mean_s = raw_12d[1]
    torso_std_v = raw_12d[5]
    return torso_mean_s < TEAM_REF_S_MAX and torso_std_v > TEAM_REF_STD_V_MIN


def normalize_to_8d(raw_12d: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    mean_h_a, mean_s_a, mean_v_a = raw_12d[0], raw_12d[1], raw_12d[2]
    mean_h_b, mean_s_b, mean_v_b = raw_12d[6], raw_12d[7], raw_12d[8]

    hx_a = np.cos(mean_h_a * np.pi / 90) * W_H
    hy_a = np.sin(mean_h_a * np.pi / 90) * W_H
    s_norm_a = (mean_s_a - scaler.mean_[0]) / scaler.scale_[0]
    v_norm_a = (mean_v_a - scaler.mean_[1]) / scaler.scale_[1]

    hx_b = np.cos(mean_h_b * np.pi / 90) * W_H * W_B
    hy_b = np.sin(mean_h_b * np.pi / 90) * W_H * W_B
    s_norm_b = ((mean_s_b - scaler.mean_[4]) / scaler.scale_[4]) * W_B
    v_norm_b = ((mean_v_b - scaler.mean_[5]) / scaler.scale_[5]) * W_B

    return np.array([hx_a, hy_a, s_norm_a, v_norm_a, hx_b, hy_b, s_norm_b, v_norm_b])


def drop_garbage_cluster(labels: np.ndarray, confidences: np.ndarray) -> tuple[int, int, int]:
    cluster_scores = []
    for k in range(3):
        mask = labels == k
        count = mask.sum()
        avg_conf = confidences[mask].mean() if count > 0 else 0.0
        cluster_scores.append((k, count * avg_conf))
    garbage_idx = min(cluster_scores, key=lambda x: x[1])[0]
    remaining = [i for i in range(3) if i != garbage_idx]
    sizes = [(labels == k).sum() for k in range(3)]
    return remaining[0], remaining[1], garbage_idx, sizes[garbage_idx]


def classify_player(
    feat_8d: np.ndarray, centroid_team0: np.ndarray, centroid_team1: np.ndarray
) -> tuple[int, float, float]:
    dist0 = float(np.linalg.norm(feat_8d - centroid_team0))
    dist1 = float(np.linalg.norm(feat_8d - centroid_team1))
    label = 0 if dist0 < dist1 else 1
    return label, dist0, dist1


class TemporalVoter:
    def __init__(self) -> None:
        self.histories: dict[int, deque] = defaultdict(lambda: deque(maxlen=DEQUE_LENGTH))
        self.locked_team: dict[int, int] = {}
        self.lock_streak: dict[int, int] = defaultdict(int)
        self.esc_streak: dict[int, int] = defaultdict(int)
        self.last_label: dict[int, int] = {}

    def update(self, track_id: int, label: int, dist0: float, dist1: float) -> int:
        margin = abs(dist0 - dist1)
        is_confident = margin >= CONFIDENCE_MARGIN

        if is_confident:
            self.histories[track_id].append(label)
            self.last_label[track_id] = label

        if track_id in self.locked_team:
            locked = self.locked_team[track_id]
            if is_confident and label != locked and margin >= ESCAPE_MARGIN:
                self.esc_streak[track_id] += 1
                self.lock_streak[track_id] = 0
                if self.esc_streak[track_id] >= ESCAPE_THRESHOLD:
                    del self.locked_team[track_id]
                    self.esc_streak[track_id] = 0
                    self.lock_streak[track_id] = 0
            else:
                self.esc_streak[track_id] = 0
            return self.locked_team.get(track_id, self._majority_vote(track_id))

        if is_confident:
            hist = self.histories[track_id]
            if len(hist) >= SOFT_LOCK_THRESHOLD:
                recent = list(hist)[-SOFT_LOCK_THRESHOLD:]
                if len(set(recent)) == 1:
                    self.locked_team[track_id] = recent[0]
                    self.lock_streak[track_id] = SOFT_LOCK_THRESHOLD

        voted = self._majority_vote(track_id)
        if voted >= 0:
            self.last_label[track_id] = voted
        return voted

    def _majority_vote(self, track_id: int) -> int:
        hist = self.histories[track_id]
        if not hist:
            return self.last_label.get(track_id, -1)
        return int(np.round(np.mean(hist)))


class FootballTeamClassifier:
    STATE_WARMUP = "WARMUP"
    STATE_CALIBRATING = "CALIBRATING"
    STATE_LOCKED = "LOCKED"

    def __init__(self) -> None:
        self.state = self.STATE_WARMUP
        self.warmup_vectors: list[np.ndarray] = []
        self.warmup_confs: list[float] = []
        self.scaler: StandardScaler | None = None
        self.centroid_team0: np.ndarray | None = None
        self.centroid_team1: np.ndarray | None = None
        self.voter = TemporalVoter()
        self.field_mask: np.ndarray | None = None
        self.cal_log = CalibrationLog()
        self.cal_fail_count = 0
        self.centroid_threshold = MINIMUM_CENTROID_DISTANCE
        self.preview_centroids: tuple[np.ndarray, np.ndarray] | None = None
        self._calibrating_this_frame = False

        # Optional HSV bounds for warmup field gate (auto-tuned)
        self.field_hsv: tuple[int, int, int, int] | None = None

    def load_calibration(self, scaler, c0, c1) -> None:
        self.scaler = scaler
        self.centroid_team0 = c0
        self.centroid_team1 = c1
        self.state = self.STATE_LOCKED
        self.cal_log.events.append("Loaded saved calibration")

    def process_frame(
        self, frame_bgr: np.ndarray, detections: list[dict]
    ) -> dict[int, int]:
        results: dict[int, int] = {}
        self._calibrating_this_frame = False

        for det in detections:
            track_id = det["track_id"]
            mask = det["mask"]
            bbox = det["bbox"]
            conf = det["conf"]

            raw_12d = build_raw_12d_vector(frame_bgr, mask, bbox)
            if raw_12d is None:
                results[track_id] = -1
                continue

            if is_referee(raw_12d):
                results[track_id] = -2
                continue

            if self.state == self.STATE_WARMUP:
                if self._passes_quality_gate(det, mask):
                    self.warmup_vectors.append(raw_12d)
                    self.warmup_confs.append(conf)
                    if len(self.warmup_vectors) >= QUALITY_FRAMES:
                        self._calibrate()
                results[track_id] = self._preview_label(raw_12d, track_id)

            elif self.state == self.STATE_CALIBRATING:
                results[track_id] = -1

            elif self.state == self.STATE_LOCKED:
                assert self.scaler is not None
                assert self.centroid_team0 is not None
                assert self.centroid_team1 is not None
                feat_8d = normalize_to_8d(raw_12d, self.scaler)
                label, dist0, dist1 = classify_player(
                    feat_8d, self.centroid_team0, self.centroid_team1
                )
                results[track_id] = self.voter.update(track_id, label, dist0, dist1)

        return results

    def _preview_label(self, raw_12d: np.ndarray, track_id: int) -> int:
        if self.preview_centroids is None or self.scaler is None:
            return -1
        feat = normalize_to_8d(raw_12d, self.scaler)
        label, _, _ = classify_player(feat, self.preview_centroids[0], self.preview_centroids[1])
        return label

    def _passes_quality_gate(self, det: dict, mask: np.ndarray) -> bool:
        if det["conf"] < WARMUP_CONF_MIN:
            return False
        if np.count_nonzero(mask) < 300:
            return False
        if self.field_mask is not None:
            x1, y1, x2, y2 = det["bbox"]
            cy = (y1 + y2) // 2
            cx = (x1 + x2) // 2
            h, w = self.field_mask.shape[:2]
            if not (0 <= cy < h and 0 <= cx < w and self.field_mask[cy, cx] > 0):
                return False
        return True

    def _calibrate(self) -> None:
        if self._calibrating_this_frame:
            return
        self._calibrating_this_frame = True
        self.state = self.STATE_CALIBRATING

        vecs = np.stack(self.warmup_vectors[-QUALITY_FRAMES * 2 :])
        confs = np.array(self.warmup_confs[-QUALITY_FRAMES * 2 :])

        self.cal_log.on_calibrate_start(len(vecs))

        self.scaler = StandardScaler()
        self.scaler.fit(vecs[:, SCALER_SV_COLS])

        normed = np.stack([normalize_to_8d(v, self.scaler) for v in vecs])
        km = KMeans(n_clusters=3, n_init=10, random_state=42)
        km.fit(normed)

        t0_idx, t1_idx, garbage_idx, garbage_n = drop_garbage_cluster(km.labels_, confs)
        c0 = km.cluster_centers_[t0_idx]
        c1 = km.cluster_centers_[t1_idx]
        dist = float(np.linalg.norm(c0 - c1))

        self.preview_centroids = (c0, c1)

        if dist <= self.centroid_threshold:
            self.cal_fail_count += 1
            self.cal_log.on_calibrate_fail(dist, self.centroid_threshold, self.cal_fail_count)
            print(self.cal_log.last_fail_reason)

            self.warmup_vectors = self.warmup_vectors[-QUALITY_FRAMES:]
            self.warmup_confs = self.warmup_confs[-QUALITY_FRAMES:]

            if self.cal_fail_count >= CALIBRATION_FAIL_MAX:
                self.centroid_threshold = MINIMUM_CENTROID_DISTANCE_FALLBACK
                print(f"Lowering centroid threshold to {self.centroid_threshold}")

            self.state = self.STATE_WARMUP
            return

        self.centroid_team0 = c0
        self.centroid_team1 = c1
        self.state = self.STATE_LOCKED
        c0_n = int((km.labels_ == t0_idx).sum())
        c1_n = int((km.labels_ == t1_idx).sum())
        self.cal_log.on_calibrate_success(dist, c0_n, c1_n, garbage_n)
        print(self.cal_log.events[-1])

    @property
    def warmup_count(self) -> int:
        return len(self.warmup_vectors)
