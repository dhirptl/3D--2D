"""Unit tests for optimization helpers."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.color_features import (
    build_raw_lab4d_vector,
    preprocess_lab_clahe,
    subtract_turf_from_mask,
    torso_mask_from_keypoints,
    upper_body_mask,
)
from src.config import (
    FEATURE_VERSION,
    SCALER_LAB_COLS,
    SEG_MODEL_PATH,
    UPPER_BODY_FRAC,
    WARMUP_VECTOR_CAP,
)
from src.field_registration import FieldRegistration
from src.helmet_verify import HelmetTrackGate, head_roi, helmet_confirms_player
from src.mask_utils import erode_mask, iou_matrix_xyxy, mask_area, match_indices
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
    assert vec.shape == (4,)


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
    kpts[5] = [20, 20, 1.0]
    kpts[6] = [40, 20, 1.0]
    kpts[11] = [20, 70, 1.0]
    kpts[12] = [40, 70, 1.0]
    torso = torso_mask_from_keypoints(crop, kpts)
    assert mask_area(torso) > 0
    assert torso[0:15, :].sum() == 0


def test_mask_area():
    m = np.zeros((10, 10), dtype=np.uint8)
    m[2:5, 2:5] = 255
    assert mask_area(m) == 9


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


def test_helmet_track_gate_requires_majority():
    gate = HelmetTrackGate(window=3, required=2, grace=0)
    assert not gate.observe(7, False)
    assert not gate.observe(7, True)
    assert gate.observe(7, True)


def test_helmet_track_gate_grace():
    gate = HelmetTrackGate(window=3, required=2, grace=1)
    assert gate.observe(9, False)
    assert not gate.observe(9, False)


def test_pipeline_context_helmet_defaults():
    ctx = VideoPipelineContext(model=None)
    assert not ctx.require_helmet
    assert ctx.helmet_model_path is None
    assert ctx.helmet_conf > 0
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
    vecs = np.random.rand(5, 4).astype(np.float64) * 50 + 100
    scaler = StandardScaler()
    scaler.fit(vecs[:, SCALER_LAB_COLS])
    out = normalize_batch_features(vecs, scaler)
    assert out.shape == (5, 4)


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
    ref = np.array([128, 128, 20, 25], dtype=float)
    team = np.array([200, 90, 10, 12], dtype=float)
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
    assert kept_xyxy.shape == (1, 4)
    assert kept_confs.shape == (1,)
    assert kept_ids.tolist() == [3]


def test_warmup_per_track_cap():
    clf = FootballTeamClassifier()
    clf.field_hsv_ready = False
    raw = np.array([200, 90, 10, 12], dtype=float)
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
    raw = np.array([200, 90, 10, 12], dtype=float)
    mask = np.ones((50, 30), dtype=np.uint8) * 255
    clf.warmup_vectors = [raw.copy() for _ in range(WARMUP_VECTOR_CAP)]
    clf.warmup_scores = [1.0] * 99 + [0.1]
    clf.warmup_track_ids = list(range(100))
    for tid in range(100):
        clf.warmup_counts_by_track[tid] = 1
    det = {"conf": 0.9, "bbox": (10, 10, 40, 60)}
    clf._try_add_warmup_sample(200, det, mask, raw, 0.95)
    assert 0.1 not in clf.warmup_scores
    assert len(clf.warmup_vectors) == 100


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
    team_a = np.array([180, 90, 8, 10], dtype=float)
    team_b = np.array([90, 180, 9, 11], dtype=float)
    for i in range(55):
        tid = i % 15
        clf.warmup_vectors.append(team_a if tid % 2 == 0 else team_b)
        clf.warmup_scores.append(1.0)
        clf.warmup_track_ids.append(tid)
        clf.warmup_counts_by_track[tid] += 1
    success = clf._calibrate()
    assert success
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


def test_feature_version():
    assert FEATURE_VERSION == "lab4d_v1"


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
    ctx = VideoPipelineContext(model=model, classifier=classifier, use_pose=False)
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
    test_helmet_track_gate_requires_majority()
    test_helmet_track_gate_grace()
    test_pipeline_context_helmet_defaults()
    test_majority_vote_counter()
    test_normalize_batch_features()
    test_prefilter_drops_neutral()
    test_is_referee_lab()
    test_adaptive_margin()
    test_field_overlap_gate()
    test_adaptive_hud_filter_uses_field_mask()
    test_warmup_per_track_cap()
    test_quality_eviction_drops_worst()
    test_load_calibration_version_mismatch()
    test_classifier_locks_on_synthetic_vectors()
    test_field_registration_synthetic()
    test_feature_version()
    test_full_pipeline_synthetic()
    print("All tests passed.")
