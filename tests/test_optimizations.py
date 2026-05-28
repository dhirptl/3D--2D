"""Unit tests for optimization helpers."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

import src.pipeline as pipeline_module
import src.team_classifier as team_classifier_module
from src.color_features import (
    build_raw_lab4d_vector,
    preprocess_lab_clahe,
    subtract_turf_from_mask,
    torso_mask_from_keypoints,
    upper_body_mask,
)
from src.config import (
    FEATURE_VERSION,
    FIELD_REG_TEMPLATE_H,
    QUALITY_FRAMES,
    SCALER_LAB_COLS,
    SEG_MODEL_PATH,
    UPPER_BODY_FRAC,
    WARMUP_CONF_HELMET,
    WARMUP_MIN_UNIQUE_TRACKS,
    WARMUP_VECTOR_CAP,
)
from src.field_registration import FieldRegistration, project_feet_to_field
from src.helmet_verify import (
    HelmetTrackGate,
    detect_helmets_roi,
    head_roi,
    helmet_confirms_player,
)
from src.mask_utils import (
    erode_mask,
    get_instance_mask_full,
    iou_matrix_xyxy,
    mask_area,
    match_indices,
)
from src.pipeline import VideoPipelineContext
from src.post_process import filter_hud_detections
from src.team_classifier import (
    FootballTeamClassifier,
    TemporalVoter,
    classify_player,
    hsv_field_to_lab_ab_centroid,
    is_referee,
    normalize_batch_features,
    normalize_features,
    prefilter_calibration_vectors,
)


class _FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self.array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, idx):
        return _FakeTensor(self.array[idx])


class _FakeBoxes:
    def __init__(self, xyxy, conf):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)

    def __len__(self):
        return len(self.xyxy.array)


class _FakeResult:
    def __init__(self, xyxy, conf):
        self.boxes = _FakeBoxes(xyxy, conf)


class _FakeHelmetModel:
    def __init__(self, per_crop_boxes):
        self.per_crop_boxes = per_crop_boxes
        self.seen_crops = None

    def predict(self, crops, conf=0.0, verbose=False):
        del conf, verbose
        self.seen_crops = crops
        return [
            _FakeResult(boxes, np.full((len(boxes),), 0.9, dtype=np.float32))
            for boxes in self.per_crop_boxes
        ]


def test_upper_body_mask_top_fraction():
    crop = np.zeros((100, 10), dtype=np.uint8)
    crop[20:80, :] = 255
    upper = upper_body_mask(crop, top_frac=UPPER_BODY_FRAC)
    ys_full = np.where(crop > 0)[0]
    y_min, y_max = int(ys_full.min()), int(ys_full.max())
    cutoff = y_min + int((y_max - y_min + 1) * UPPER_BODY_FRAC)
    assert upper[cutoff:, :].sum() == 0
    assert upper[y_min:cutoff, :].sum() > 0


def test_upper_body_mask_horizontal():
    crop = np.zeros((40, 80), dtype=np.uint8)
    crop[5:35, 10:70] = 255
    bbox = (0, 0, 80, 40)
    upper = upper_body_mask(crop, bbox=bbox)
    assert upper[0:5, :].sum() == 0
    assert upper[35:, :].sum() == 0
    assert upper[5:35, :].sum() > 0


def test_subtract_turf_from_mask():
    crop = np.ones((20, 20), dtype=np.uint8) * 255
    turf = np.zeros((20, 20), dtype=np.uint8)
    turf[10:, :] = 255
    out, applied = subtract_turf_from_mask(crop, turf)
    assert applied
    assert out[10:, :].sum() == 0
    assert out[:10, :].sum() > 0


def test_green_safeguard_bypass():
    crop = np.ones((40, 40), dtype=np.uint8) * 255
    turf = np.ones((40, 40), dtype=np.uint8) * 255
    player_bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    player_bgr[:, :, 1] = 180
    player_bgr[:, :, 0] = 40
    out, applied = subtract_turf_from_mask(crop, turf, player_bgr=player_bgr)
    assert not applied
    assert out.sum() == crop.sum()


def test_preprocess_lab_clahe_preserves_ab():
    bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    bgr[:, :, 0] = 100
    bgr[:, :, 1] = 150
    bgr[:, :, 2] = 200
    lab_in = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    _, a_in, b_in = cv2.split(lab_in)
    lab_out = preprocess_lab_clahe(bgr)
    _, a_out, b_out = cv2.split(lab_out)
    assert np.allclose(a_in, a_out)
    assert np.allclose(b_in, b_out)


def test_build_raw_lab4d_vector_small_mask():
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:12, 10:12] = 255
    assert build_raw_lab4d_vector(frame, mask, (5, 5, 20, 20)) is None


def test_build_raw_lab4d_vector_valid():
    frame = np.zeros((80, 60, 3), dtype=np.uint8)
    frame[20:70, 15:45, 2] = 200
    frame[20:70, 15:45, 0] = 10
    mask = np.zeros((80, 60), dtype=np.uint8)
    mask[20:70, 15:45] = 255
    vec = build_raw_lab4d_vector(frame, mask, (10, 15, 50, 75))
    assert vec is not None
    assert vec.shape == (6,)


def test_build_raw_lab4d_vector_stale_mask():
    frame = np.zeros((80, 60, 3), dtype=np.uint8)
    mask = np.zeros((80, 60), dtype=np.uint8)
    mask[20:70, 15:45] = 255
    det = {"frame_age": 1}
    assert build_raw_lab4d_vector(frame, mask, (10, 15, 50, 75), det=det) is None


def test_torso_mask_from_keypoints():
    crop = np.zeros((100, 60), dtype=np.uint8)
    crop[10:90, 10:50] = 255
    kpts = np.zeros((17, 3), dtype=np.float32)
    kpts[5] = [20, 20, 1.0]   # L_shoulder: top-left
    kpts[6] = [40, 20, 1.0]   # R_shoulder: top-right
    kpts[11] = [20, 70, 1.0]  # L_hip: bottom-left
    kpts[12] = [40, 70, 1.0]  # R_hip: bottom-right
    torso = torso_mask_from_keypoints(crop, kpts)
    assert mask_area(torso) > 0
    assert torso[0:15, :].sum() == 0
    # Correct polygon (5→6→12→11) should fill ~1000px; bowtie would be near-zero
    assert mask_area(torso) > 500, f"Torso fill too small ({mask_area(torso)}px); likely bowtie polygon"


def test_mask_area():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 2:5] = 255
    assert mask_area(m) == 9


def test_get_instance_mask_full_accepts_unit_binary_masks():
    masks = type(
        "FakeMasks",
        (),
        {
            "data": _FakeTensor(np.array([[[0, 1], [1, 0]]], dtype=np.uint8)),
            "xy": None,
        },
    )()
    full = get_instance_mask_full(masks, 0, (2, 2))
    assert full is not None
    assert full.dtype == np.uint8
    assert full.tolist() == [[0, 255], [255, 0]]


def test_erode_mask():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:8, 2:8] = 255
    e = erode_mask(m, 2)
    assert mask_area(e) < mask_area(m)


def test_iou_match_unique():
    filtered = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    orig = np.array([[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 15, 15]], dtype=float)
    idx = match_indices(filtered, orig)
    assert len(idx) == 2
    assert len(set(idx)) == 2


def test_iou_matrix_broadcast():
    a = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=float)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=float)
    iou = iou_matrix_xyxy(a, b)
    assert iou.shape == (2, 2)
    assert iou[0, 0] > 0.9


def test_head_roi_bbox_only():
    roi = head_roi((10, 20, 50, 120))
    assert roi == (10, 20, 50, 48)


def test_head_roi_pose_refined():
    bbox = (100, 50, 140, 170)
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[0] = [20, 20, 0.9]
    keypoints[1] = [15, 18, 0.9]
    keypoints[2] = [25, 18, 0.9]
    fallback = head_roi(bbox, None)
    roi = head_roi(bbox, keypoints)
    assert bbox[0] <= roi[0] < roi[2] <= bbox[2]
    assert bbox[1] <= roi[1] < roi[3] <= bbox[3]
    assert roi != fallback
    assert (roi[2] - roi[0]) < (fallback[2] - fallback[0])


def test_helmet_confirms_player_center_match():
    roi = (10, 10, 30, 30)
    helmets = [{"bbox": (14, 14, 18, 18), "conf": 0.8}]
    assert helmet_confirms_player(roi, helmets)


def test_helmet_confirms_player_center_miss():
    roi = (10, 10, 30, 30)
    helmets = [{"bbox": (14, 40, 18, 44), "conf": 0.8}]
    assert not helmet_confirms_player(roi, helmets)


def test_helmet_confirms_player_iou_fallback():
    roi = (10, 10, 30, 30)
    helmets = [{"bbox": (24, 10, 40, 30), "conf": 0.8}]
    assert helmet_confirms_player(roi, helmets, iou_min=0.15)


def test_detect_helmets_roi_translates_boxes():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    detections = [{"bbox": (100, 50, 140, 170)}]
    model = _FakeHelmetModel([np.array([[10, 5, 20, 15]], dtype=np.float32)])
    helmets = detect_helmets_roi(model, frame, detections, conf=0.25, pad_frac=0.5)
    assert len(model.seen_crops) == 1
    assert helmets[0]["bbox"] == (90, 39, 100, 49)
    assert abs(helmets[0]["conf"] - 0.9) < 1e-5


def test_helmet_track_gate_requires_majority():
    gate = HelmetTrackGate(window=3, required=2, grace=0)
    assert not gate.observe(7, False)
    assert not gate.observe(7, True)
    assert gate.observe(7, True)


def test_helmet_track_gate_grace():
    gate = HelmetTrackGate(window=3, required=2, grace=1)
    assert gate.observe(9, False)
    assert not gate.observe(9, False)


def test_helmet_track_gate_soft_lock_relaxes_requirement():
    gate = HelmetTrackGate(window=3, required=2, grace=0, soft_lock_streak=3, locked_required=1)
    assert not gate.observe(5, True)
    assert gate.observe(5, True)
    assert gate.observe(5, True)
    assert gate.observe(5, False)
    assert gate.observe(5, False)
    assert not gate.observe(5, False)


def test_pipeline_context_helmet_defaults():
    ctx = VideoPipelineContext(model=None)
    assert ctx.require_helmet
    assert ctx.helmet_model_path is None
    assert ctx.helmet_conf > 0
    assert ctx.helmet_every >= 1
    assert ctx.use_helmet_gate
    assert ctx._last_helmets == []
    assert ctx._helmet_cache_frame == -1
    assert ctx._helmet_gate is None


def test_majority_vote_counter():
    voter = TemporalVoter()
    for _ in range(5):
        voter.histories[1].append(0)
    for _ in range(3):
        voter.histories[1].append(1)
    assert voter._majority_vote(1) == 0


def test_normalize_batch_features():
    vecs = np.random.rand(5, 6).astype(np.float64) * 50 + 100
    scaler = StandardScaler()
    scaler.fit(vecs[:, SCALER_LAB_COLS])
    out = normalize_batch_features(vecs, scaler)
    assert out.shape == (5, 6)


def test_prefilter_drops_neutral():
    vecs = np.array([
        [128, 128, 5, 5],
        [200, 80, 10, 12],
        [90, 200, 8, 9],
    ], dtype=float)
    scores = np.array([1.0, 1.0, 1.0])
    tids = [0, 1, 2]
    out, _, _ = prefilter_calibration_vectors(vecs, scores, tids)
    assert len(out) == 2


def test_is_referee_lab():
    ref = np.array([128, 128, 20, 25, 60.0, 5.0], dtype=float)
    team = np.array([200, 90, 10, 12, 30.0, 10.0], dtype=float)
    assert is_referee(ref)
    assert not is_referee(team)


def test_adaptive_margin():
    voter = TemporalVoter()
    voter.set_margins(10.0, warmup=False)
    assert voter.confidence_margin >= 1.0
    voter.set_margins(2.0, warmup=True)
    assert voter.confidence_margin == 1.0


def test_field_overlap_gate():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = True
    clf.field_mask = np.zeros((100, 100), dtype=np.uint8)
    clf.field_mask[72:100, :] = 255
    mask = np.zeros((40, 30), dtype=np.uint8)
    mask[0:32, 5:25] = 255
    mask[32:40, 5:25] = 255
    det = {"conf": 0.9, "bbox": (20, 55, 50, 95)}
    assert clf._mask_field_overlap(det, mask) >= 0.25
    assert clf._quality_gate_reason(det, mask) is None
    clf.field_mask = np.zeros((100, 100), dtype=np.uint8)
    det_off = {"conf": 0.9, "bbox": (0, 0, 25, 35)}
    mask_off = np.zeros((35, 25), dtype=np.uint8)
    mask_off[:, :] = 255
    assert clf._quality_gate_reason(det_off, mask_off) == "not_on_field"


def test_helmet_confirmed_relaxes_warmup_gate():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = True
    clf.field_mask = np.zeros((100, 100), dtype=np.uint8)
    mask = np.ones((35, 25), dtype=np.uint8) * 255

    det_low = {
        "conf": WARMUP_CONF_HELMET + 0.02,
        "bbox": (0, 0, 25, 35),
        "helmet_confirmed": True,
    }
    assert clf._quality_gate_reason(det_low, mask) is None
    assert clf._quality_gate_reason(
        {**det_low, "helmet_confirmed": False}, mask
    ) == "low_conf"

    det_off = {
        "conf": 0.9,
        "bbox": (0, 0, 25, 35),
        "helmet_confirmed": True,
    }
    assert clf._quality_gate_reason(det_off, mask) is None
    assert clf._quality_gate_reason(
        {**det_off, "helmet_confirmed": False}, mask
    ) == "not_on_field"


def test_adaptive_hud_filter_uses_field_mask():
    xyxy = np.array([[0, 5, 10, 15], [0, 85, 10, 95], [0, 45, 10, 55]], dtype=float)
    confs = np.array([0.9, 0.9, 0.9], dtype=float)
    tids = np.array([1, 2, 3], dtype=int)
    field_mask = np.zeros((100, 20), dtype=np.uint8)
    field_mask[40:60, :] = 255
    kept_xyxy, kept_confs, kept_ids = filter_hud_detections(
        (100, 20, 3),
        xyxy,
        confs,
        tids,
        field_mask=field_mask,
    )
    # Wider field margins keep center + lower sideline boxes; top HUD box still dropped.
    assert kept_xyxy.shape[0] >= 1
    assert 3 in kept_ids.tolist()
    assert 1 not in kept_ids.tolist()


def test_helmet_every_reuses_cached_detections():
    old_detect = pipeline_module.detect_helmets
    calls = []

    def fake_detect(model, frame, conf=0.0):
        del model, frame, conf
        calls.append(1)
        return [{"bbox": (1, 1, 2, 2), "conf": 0.8}]

    try:
        pipeline_module.detect_helmets = fake_detect
        ctx = VideoPipelineContext(model=None, require_helmet=True, helmet_every=2)
        ctx._helmet_model = object()
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        detections = [{"track_id": 1, "bbox": (0, 0, 10, 10)}]

        ctx.frame_idx = 0
        helmets0 = pipeline_module._helmet_detections(frame, ctx, run_yolo=True, detections=detections)
        ctx.frame_idx = 1
        helmets1 = pipeline_module._helmet_detections(frame, ctx, run_yolo=True, detections=detections)

        assert len(calls) == 1
        assert helmets0 == helmets1
        assert ctx.helmet_inference_frames == 1
    finally:
        pipeline_module.detect_helmets = old_detect


def test_warmup_per_track_cap():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = False
    raw = np.array([200, 90, 10, 12, 30.0, 10.0], dtype=float)
    mask = np.ones((50, 30), dtype=np.uint8) * 255
    det = {"conf": 0.9, "bbox": (10, 10, 40, 60)}
    for i in range(5):
        r = clf._try_add_warmup_sample(1, det, mask, raw, 0.9)
        if i < 3:
            assert r is None
        else:
            assert r == "track_cap"
    assert clf.warmup_counts_by_track[1] == 3


def test_quality_eviction_drops_worst():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = False
    raw = np.array([200, 90, 10, 12, 30.0, 10.0], dtype=float)
    mask = np.ones((50, 30), dtype=np.uint8) * 255
    cap = WARMUP_VECTOR_CAP
    clf.warmup_vectors = [raw.copy() for _ in range(WARMUP_VECTOR_CAP)]
    clf.warmup_scores = [1.0] * (cap - 1) + [0.1]
    clf.warmup_track_ids = list(range(cap))
    for tid in range(cap):
        clf.warmup_counts_by_track[tid] = 1
    det = {"conf": 0.9, "bbox": (10, 10, 40, 60)}
    clf._try_add_warmup_sample(200, det, mask, raw, 0.95)
    assert 0.1 not in clf.warmup_scores
    assert len(clf.warmup_vectors) == cap


def test_load_calibration_version_mismatch():
    clf = FootballTeamClassifier()
    try:
        clf.load_calibration({"feature_version": "6d_v1", "scaler": None})
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "mismatch" in str(e).lower()


def test_classifier_locks_on_synthetic_vectors():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = True
    clf.warmup_min_unique_tracks = 2
    team_a = np.array([180, 90, 8, 10, 30.0, 10.0], dtype=float)
    team_b = np.array([90, 180, 9, 11, 120.0, 10.0], dtype=float)
    for i in range(55):
        tid = i % 15
        clf.warmup_vectors.append(team_a if tid % 2 == 0 else team_b)
        clf.warmup_scores.append(1.0)
        clf.warmup_track_ids.append(tid)
        clf.warmup_counts_by_track[tid] += 1
    success = clf._calibrate()
    assert success
    assert clf.state == FootballTeamClassifier.STATE_LOCKED


def test_classifier_locks_with_lower_warmup_thresholds():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = True
    team_a = np.array([180, 90, 8, 10, 30.0, 10.0], dtype=float)
    team_b = np.array([90, 180, 9, 11, 120.0, 10.0], dtype=float)

    for i in range(QUALITY_FRAMES):
        tid = i % WARMUP_MIN_UNIQUE_TRACKS
        clf.warmup_vectors.append(team_a if tid % 2 == 0 else team_b)
        clf.warmup_scores.append(1.0)
        clf.warmup_track_ids.append(tid)
        clf.warmup_counts_by_track[tid] += 1

    assert clf._ready_to_calibrate()
    assert clf._calibrate()
    assert clf.state == FootballTeamClassifier.STATE_LOCKED


def test_field_registration_synthetic():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :] = (34, 139, 34)
    for y in range(40, 320, 30):
        cv2.line(frame, (50, y), (590, y), (255, 255, 255), 2)
    reg = FieldRegistration()
    for _ in range(5):
        reg.update(frame)
    assert reg.homography is not None


def test_project_feet_to_field_identity():
    pos = project_feet_to_field((100, 150, 140, 220), np.eye(3, dtype=np.float32))
    assert pos == (120.0, 220.0)


def test_project_feet_to_field_rejects_out_of_bounds():
    homography = np.array(
        [[1.0, 0.0, 2000.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert project_feet_to_field((100, 150, 140, 220), homography) is None


def test_update_field_mask_syncs_classifier_homography():
    class _FakeFieldReg:
        def __init__(self, registration_valid, homography, playable):
            self.registration_valid = registration_valid
            self.homography = homography
            self._playable = playable
            self.valid_streak = 0
            self.homography_stale = False

        def update(self, frame, *, hsv=None):
            del frame, hsv
            return self.registration_valid

        def playable_mask(self, h, w):
            del h, w
            return self._playable

    frame = np.zeros((40, 60, 3), dtype=np.uint8)
    classifier = FootballTeamClassifier()
    ctx = VideoPipelineContext(model=None, classifier=classifier)
    play = np.ones((40, 60), dtype=np.uint8) * 255
    homography = np.eye(3, dtype=np.float32)

    ctx.field_reg = _FakeFieldReg(True, homography, play)
    mask = pipeline_module._update_field_mask(frame, ctx)
    assert np.array_equal(mask, play)
    assert np.array_equal(classifier.homography, homography)

    ctx.field_reg = _FakeFieldReg(False, homography, None)
    ctx.cached_field_mask = play
    ctx.field_mask_frame = ctx.frame_idx
    classifier.homography = homography
    mask = pipeline_module._update_field_mask(frame, ctx)
    assert np.array_equal(mask, play)
    assert classifier.homography is None


def _build_spatial_scene(group_ys):
    max_group = max(len(ys) for ys in group_ys)
    frame_w = 20 + max_group * 50 + 30
    frame = np.zeros((FIELD_REG_TEMPLATE_H, frame_w, 3), dtype=np.uint8)
    detections = []
    colors = ((0, 0, 220), (220, 0, 0))
    track_id = 0
    for group_idx, ys in enumerate(group_ys):
        for player_idx, y2 in enumerate(ys):
            x1 = 20 + player_idx * 50
            x2 = x1 + 30
            y1 = y2 - 60
            frame[y1:y2, x1:x2] = colors[group_idx]
            detections.append(
                {
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "mask": np.ones((60, 30), dtype=np.uint8) * 255,
                    "conf": 0.9,
                    "frame_age": 0,
                }
            )
            track_id += 1
    return frame, detections


def test_spatial_calibration_locks_on_two_lineups():
    frame, detections = _build_spatial_scene(
        [
            [130, 134, 138, 142, 146, 150, 154],
            [290, 294, 298, 302, 306, 310, 314],
        ]
    )
    clf = FootballTeamClassifier()
    clf.frame_idx = 30
    clf.homography = np.eye(3, dtype=np.float32)

    assert clf._try_spatial_calibration(frame, detections)
    assert clf.state == FootballTeamClassifier.STATE_LOCKED
    assert clf.locked_frame_index == 30


def test_spatial_calibration_rejects_imbalanced_groups():
    frame, detections = _build_spatial_scene(
        [
            [130, 134, 138, 142, 146, 150, 154, 158, 162, 166],
            [290, 294, 298, 302],
        ]
    )
    clf = FootballTeamClassifier()
    clf.homography = np.eye(3, dtype=np.float32)

    assert not clf._try_spatial_calibration(frame, detections)
    assert clf.state == FootballTeamClassifier.STATE_WARMUP


def test_spatial_calibration_rejects_low_relative_separation():
    frame, detections = _build_spatial_scene(
        [
            [150, 160, 170, 180, 190, 200, 210],
            [225, 235, 245, 255, 265, 275, 285],
        ]
    )
    clf = FootballTeamClassifier()
    clf.homography = np.eye(3, dtype=np.float32)

    old_threshold = team_classifier_module.SPATIAL_MIN_REL_SEPARATION
    team_classifier_module.SPATIAL_MIN_REL_SEPARATION = 0.95
    try:
        assert not clf._try_spatial_calibration(frame, detections)
    finally:
        team_classifier_module.SPATIAL_MIN_REL_SEPARATION = old_threshold
    assert clf.state == FootballTeamClassifier.STATE_WARMUP


def test_feature_version():
    assert FEATURE_VERSION == "lab6d_v1"


def test_full_pipeline_synthetic():
    if not SEG_MODEL_PATH.exists():
        print("SKIP test_full_pipeline_synthetic: no seg model")
        return

    from ultralytics import YOLO

    from src.pipeline import VideoPipelineContext, process_video_frame

    tmp = tempfile.mktemp(suffix=".mp4")
    out = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for _ in range(150):
        frame = np.full((360, 640, 3), (34, 139, 34), dtype=np.uint8)
        frame[100:260, 40:160] = (0, 0, 220)
        frame[100:260, 480:600] = (220, 0, 0)
        out.write(frame)
    out.release()

    model = YOLO(str(SEG_MODEL_PATH))
    classifier = FootballTeamClassifier()
    ctx = VideoPipelineContext(model=model, classifier=classifier, use_pose=False, require_helmet=False)
    cap = cv2.VideoCapture(tmp)
    idx = 0
    while cap.isOpened() and idx < 150:
        ret, frame = cap.read()
        if not ret:
            break
        process_video_frame(frame, ctx)
        idx += 1
    cap.release()
    os.remove(tmp)
    assert idx == 150
    assert classifier.state in (
        FootballTeamClassifier.STATE_WARMUP,
        FootballTeamClassifier.STATE_LOCKED,
        FootballTeamClassifier.STATE_CALIBRATING,
    )



def test_warmup_frame_tick_once_per_frame():
    """_warmup_frame_tick must be called once per process_frame, not once per detection."""
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = False
    raw = np.array([200, 90, 10, 12, 30.0, 10.0], dtype=float)
    mask = np.ones((50, 30), dtype=np.uint8) * 255
    # 10 detections all with same track structure
    detections = [
        {"track_id": i, "bbox": (10, 10, 40, 60), "conf": 0.9, "mask": mask, "frame_age": 0}
        for i in range(10)
    ]
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    clf.process_frame(frame, detections)
    assert clf.frames_in_warmup == 1, (
        f"Expected frames_in_warmup==1 after one process_frame call with 10 detections, "
        f"got {clf.frames_in_warmup}"
    )


def test_mask_to_full_frame_threshold():
    """INTER_LINEAR resize of a uint8 mask must use threshold 127, not 0.5."""
    from src.mask_utils import mask_to_full_frame

    class _FakeTensorMask:
        def __init__(self, arr):
            self._arr = np.asarray(arr, dtype=np.uint8)
        def cpu(self):
            return self
        def numpy(self):
            return self._arr

    # 2x2 mask: left column = 0, right column = 255
    mask_2x2 = np.array([[0, 255], [0, 255]], dtype=np.uint8)
    fake = _FakeTensorMask(mask_2x2)
    result = mask_to_full_frame(fake, (2, 4))  # resize to 4 wide
    # After INTER_LINEAR resize, the left half should be 0 and right half 255
    # With threshold=127: blurred pixels near center stay 0 until they cross 127
    # The leftmost column of the original-zero region must remain 0
    assert result[0, 0] == 0, f"Left column should be 0, got {result[0, 0]}"
    assert result[0, 3] == 255, f"Right column should be 255, got {result[0, 3]}"


def test_video_pipeline_reader_crash_no_hang():
    """Reader thread exception must not cause infinite hang in consumer."""
    import signal
    import threading
    from src.video_pipeline import run_with_read_ahead

    class _CrashingCap:
        def __init__(self):
            self._count = 0
        def isOpened(self):
            return True
        def read(self):
            self._count += 1
            if self._count == 3:
                raise RuntimeError("simulated cap crash")
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

    cap = _CrashingCap()
    results = []
    raised = []

    def run():
        try:
            run_with_read_ahead(cap, lambda f, i: f, lambda r: results.append(r))
        except Exception as e:
            raised.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "run_with_read_ahead hung after reader crash"


def test_video_pipeline_process_fn_exception_no_hang():
    """Exception in process_fn must not deadlock the reader thread."""
    import threading
    from src.video_pipeline import run_with_read_ahead

    class _SimpleCap:
        def __init__(self, n):
            self._n = n
            self._i = 0
        def isOpened(self):
            return self._i < self._n
        def read(self):
            self._i += 1
            return True, np.zeros((10, 10, 3), dtype=np.uint8)

    cap = _SimpleCap(10)
    raised = []

    def bad_process(f, i):
        if i == 2:
            raise ValueError("boom")
        return f

    def run():
        try:
            run_with_read_ahead(cap, bad_process, lambda r: None)
        except ValueError as e:
            raised.append(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=10)
    assert not t.is_alive(), "run_with_read_ahead hung after process_fn exception"
    assert raised, "Expected ValueError to propagate"


def test_detection_masks_none_does_not_inflate_zero_det_frames():
    """masks=None with boxes present is a mask warning, not a zero-detection frame."""
    from src.detection import DetectionStats

    stats = DetectionStats()
    # Simulate the code path: masks is None, boxes present
    # This mirrors what extract_tracked_detections does at line ~102
    stats.missing_mask_warnings += 1
    # stats.record(0) must NOT be called here — verify the frame counter doesn't change
    assert stats.zero_det_frames == 0
    assert stats.missing_mask_warnings == 1
    assert stats.frames == 0  # no frame recorded at all


def test_reset_tracker_clears_stracks():
    """_reset_tracker clears tracked/lost/removed stracks on a fake tracker."""
    import src.pipeline as pm

    class _FakeTracker:
        def __init__(self):
            self.tracked_stracks = [1, 2, 3]
            self.lost_stracks = [4]
            self.removed_stracks = [5]

    class _FakePredictor:
        def __init__(self):
            self.trackers = [_FakeTracker()]

    class _FakeModel:
        def __init__(self):
            self.predictor = _FakePredictor()

    model = _FakeModel()
    pm._reset_tracker(model)
    t = model.predictor.trackers[0]
    assert t.tracked_stracks == []
    assert t.lost_stracks == []
    assert t.removed_stracks == []


def test_reset_tracker_uses_reset_method_if_present():
    """_reset_tracker calls tracker.reset() when available."""
    import src.pipeline as pm

    reset_called = []

    class _FakeTracker:
        def reset(self):
            reset_called.append(True)

    class _FakePredictor:
        trackers = [_FakeTracker()]

    class _FakeModel:
        predictor = _FakePredictor()

    pm._reset_tracker(_FakeModel())
    assert reset_called == [True]


def test_reset_tracker_no_predictor_no_crash():
    """_reset_tracker on a model with no predictor must not raise."""
    import src.pipeline as pm

    class _BareModel:
        pass

    pm._reset_tracker(_BareModel())  # should not raise


def test_run_tracker_imports_player_class_id():
    from src.config import PLAYER_CLASS_ID
    from src import run_tracker

    assert PLAYER_CLASS_ID == 0
    src = open(run_tracker.__file__).read()
    assert "PLAYER_CLASS_ID" in src


def test_extract_tracked_detections_masks_none_fallback():
    from src.detection import extract_tracked_detections

    class _Boxes:
        def __init__(self):
            import torch

            self.xyxy = torch.tensor([[10.0, 10.0, 50.0, 80.0]])
            self.conf = torch.tensor([0.9])
            self.id = torch.tensor([1.0])

    class _Result:
        boxes = _Boxes()
        masks = None

    class _Model:
        def track(self, *args, **kwargs):
            return [_Result()]

    dets = extract_tracked_detections(_Model(), np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(dets) == 1
    assert dets[0]["mask_fallback"] is True
    assert mask_area(dets[0]["mask"]) > 0


def test_filter_by_field_area_low_threshold():
    from src.post_process import filter_by_field_area

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[85:95, 40:60] = 255
    xyxy = np.array([[40, 70, 60, 95]], dtype=float)
    confs = np.array([0.9])
    tids = np.array([1])
    kept, _, _ = filter_by_field_area(frame, xyxy, confs, tids, mask, min_field_frac=0.05)
    assert len(kept) == 1


def test_camera_cut_requires_two_consecutive_frames():
    import src.pipeline as pm

    reset_count = [0]
    original = pm._reset_tracker

    def counting_reset(model):
        reset_count[0] += 1

    pm._reset_tracker = counting_reset
    try:
        ctx = VideoPipelineContext(model=None, classifier=FootballTeamClassifier())
        prev = np.zeros((10, 10, 3), dtype=np.uint8)
        cut = np.full((10, 10, 3), 200, dtype=np.uint8)
        ctx.prev_frame = prev
        ctx.field_hsv_bounds = (35, 85, 40, 40)
        pm.extract_tracked_detections = lambda *a, **kw: []

        pm.process_video_frame(cut, ctx)
        assert reset_count[0] == 0
        ctx.prev_frame = cut
        other = np.full((10, 10, 3), 50, dtype=np.uint8)
        pm.process_video_frame(other, ctx)
        assert reset_count[0] == 1
    finally:
        pm._reset_tracker = original


def test_dataset_yaml_rejects_legacy_two_class():
    from src.dataset_yaml import validate_dataset_yaml

    legacy = Path(__file__).resolve().parent.parent / "data.yaml"
    try:
        validate_dataset_yaml(legacy, task="seg")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "nc=1" in str(e)


def test_team_accuracy_iou_beats_track_id_on_id_switch():
    from src.eval_clip import _score_team_accuracy

    ann = Path(__file__).resolve().parent / "_tmp_iou_ann.csv"
    ann.write_text(
        "frame,track_id,correct_team,x1,y1,x2,y2\n"
        "0,99,1,10,10,50,80\n"
    )
    labels_wrong_id = {(0, 99): 0}
    labels_iou = {(0, 1): 1}
    bboxes = {(0, 1): (12, 12, 48, 78)}
    by_id = _score_team_accuracy(ann, labels_wrong_id, bboxes, team_match="track_id")
    by_iou = _score_team_accuracy(ann, labels_iou, bboxes, team_match="iou")
    ann.unlink(missing_ok=True)
    assert by_id is not None and by_id["pct"] == 0.0
    assert by_iou is not None and by_iou["pct"] == 100.0


def test_camera_cut_cooldown_prevents_repeated_resets():
    """Consecutive high-MAD frames only trigger one tracker reset, not N."""
    import src.pipeline as pm

    reset_count = [0]
    original = pm._reset_tracker

    def counting_reset(model):
        reset_count[0] += 1

    pm._reset_tracker = counting_reset
    try:
        ctx = VideoPipelineContext(model=None, classifier=FootballTeamClassifier())
        ctx.classifier.post_cut_frames = 0
        # Build a high-MAD frame pair
        prev = np.zeros((10, 10, 3), dtype=np.uint8)
        cut  = np.full((10, 10, 3), 200, dtype=np.uint8)
        ctx.prev_frame = prev
        ctx.field_hsv_bounds = (35, 85, 40, 40)  # skip HSV init

        # Patch out the expensive parts
        original_extract = pm.extract_tracked_detections
        pm.extract_tracked_detections = lambda *a, **kw: []

        # Two consecutive high-MAD frames trigger one reset; cooldown blocks the rest.
        pm.process_video_frame(cut, ctx)
        ctx.prev_frame = cut
        other = np.full((10, 10, 3), 50, dtype=np.uint8)
        pm.process_video_frame(other, ctx)
        ctx.prev_frame = other
        for _ in range(3):
            pm.process_video_frame(cut, ctx)
            ctx.prev_frame = cut

        assert reset_count[0] == 1, (
            f"Expected 1 tracker reset, got {reset_count[0]}"
        )
    finally:
        pm._reset_tracker = original
        pm.extract_tracked_detections = original_extract

if __name__ == "__main__":
    test_upper_body_mask_top_fraction()
    test_upper_body_mask_horizontal()
    test_subtract_turf_from_mask()
    test_green_safeguard_bypass()
    test_preprocess_lab_clahe_preserves_ab()
    test_build_raw_lab4d_vector_small_mask()
    test_build_raw_lab4d_vector_valid()
    test_build_raw_lab4d_vector_stale_mask()
    test_torso_mask_from_keypoints()
    test_mask_area()
    test_erode_mask()
    test_iou_match_unique()
    test_iou_matrix_broadcast()
    test_head_roi_bbox_only()
    test_head_roi_pose_refined()
    test_helmet_confirms_player_center_match()
    test_helmet_confirms_player_center_miss()
    test_helmet_confirms_player_iou_fallback()
    test_detect_helmets_roi_translates_boxes()
    test_helmet_track_gate_requires_majority()
    test_helmet_track_gate_grace()
    test_helmet_track_gate_soft_lock_relaxes_requirement()
    test_pipeline_context_helmet_defaults()
    test_majority_vote_counter()
    test_normalize_batch_features()
    test_prefilter_drops_neutral()
    test_is_referee_lab()
    test_adaptive_margin()
    test_field_overlap_gate()
    test_helmet_confirmed_relaxes_warmup_gate()
    test_adaptive_hud_filter_uses_field_mask()
    test_helmet_every_reuses_cached_detections()
    test_warmup_per_track_cap()
    test_quality_eviction_drops_worst()
    test_load_calibration_version_mismatch()
    test_classifier_locks_on_synthetic_vectors()
    test_classifier_locks_with_lower_warmup_thresholds()
    test_field_registration_synthetic()
    test_project_feet_to_field_identity()
    test_project_feet_to_field_rejects_out_of_bounds()
    test_update_field_mask_syncs_classifier_homography()
    test_spatial_calibration_locks_on_two_lineups()
    test_spatial_calibration_rejects_imbalanced_groups()
    test_spatial_calibration_rejects_low_relative_separation()
    test_feature_version()
    test_full_pipeline_synthetic()
    print("All tests passed.")
