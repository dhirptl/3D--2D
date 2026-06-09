"""Shared per-frame video processing for run_tracker and eval_clip."""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    BBOX_EMA_ALPHA,
    CAMERA_CUT_CONSECUTIVE_FRAMES,
    CAMERA_CUT_MAD_THRESHOLD,
    FIELD_FOOT_MIN_FRAC,
    PLAYER_DEVICE,
    PLAYER_PREDICT_HALF,
    FIELD_HSV_AUTO_FRAMES,
    FIELD_HSV_BLEND_ALPHA,
    FIELD_HSV_CALIB_PX_PER_FRAME,
    FIELD_HSV_UPDATE_INTERVAL,
    FIELD_MASK_INTERVAL,
    FIELD_REG_STABLE_STREAK,
    FIELD_REG_UPDATE_STRIDE_STABLE,
    FIELD_REG_VALID_STREAK,
    HUD_BAND_HOLD_FRAMES,
    FIELD_ROI_X0_FRAC,
    FIELD_ROI_X1_FRAC,
    FIELD_ROI_Y0_FRAC,
    FIELD_ROI_Y1_FRAC,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    POSE_EVERY_DEFAULT,
    POST_CUT_FRAMES,
)
from src.detection import DetectionStats, extract_tracked_detections
from src.field_hsv import build_field_mask_from_bounds, estimate_field_hsv, estimate_hsv_from_pixel_arrays
from src.field_registration import FieldRegistration, center_roi_green_fraction
from src.pose_estimation import attach_keypoints_to_detections, get_pose_model, run_pose_on_frame
from src.post_process import build_field_mask, filter_by_field_area, hold_hud_band_limits, suppress_duplicate_tracks, _hud_limits
from src.team_classifier import FootballTeamClassifier
from src.track_reassoc import get_reassociator


@dataclass
class VideoPipelineContext:
    model: YOLO
    classifier: FootballTeamClassifier | None = None
    det_stats: DetectionStats = field(default_factory=DetectionStats)
    apply_hud_filter: bool = True
    filter_field: bool = False
    detect_every: int = 1
    retina_masks: bool = False
    tracker_cfg: Path | None = None
    player_conf: float = PLAYER_PREDICT_CONF
    player_iou: float = PLAYER_PREDICT_IOU
    player_imgsz: int = PLAYER_IMGSZ
    player_max_det: int = PLAYER_PREDICT_MAX_DET
    player_half: bool = PLAYER_PREDICT_HALF
    player_device: str | None = None
    pose_every: int = POSE_EVERY_DEFAULT
    use_pose: bool = True
    field_hsv_bounds: tuple[int, int, int, int] | None = None
    auto_frames: list = field(default_factory=list)
    last_detections: list = field(default_factory=list)
    last_team_labels: dict[int, int] = field(default_factory=dict)
    frame_idx: int = 0
    prev_frame: np.ndarray | None = None
    field_update_counter: int = 0
    field_hsv_updated_this_frame: bool = False
    cached_field_mask: np.ndarray | None = None
    field_mask_frame: int = -1
    field_reg: FieldRegistration = field(default_factory=FieldRegistration)
    registration_valid_frames: int = 0
    hsv_deferred: bool = True
    _pose_model: YOLO | None = None
    track_id_changes: int = 0
    prev_track_ids: set[int] = field(default_factory=set)
    camera_cut_cooldown: int = 0
    camera_cut_streak: int = 0
    hud_rejected_total: int = 0
    hud_band_history: list[tuple[float, float]] = field(default_factory=list)
    field_rejected_total: int = 0
    debug_filters: bool = False
    bbox_smoother: dict = field(default_factory=dict)
    last_gray: np.ndarray | None = None


def _blend_hsv_bounds(
    old: tuple[int, int, int, int], new: tuple[int, int, int, int], alpha: float
) -> tuple[int, int, int, int]:
    return tuple(int((1 - alpha) * o + alpha * n) for o, n in zip(old, new))


def _update_field_hsv_rolling(
    frame: np.ndarray, ctx: VideoPipelineContext, hsv: np.ndarray
) -> None:
    if ctx.field_hsv_bounds is None:
        return
    ctx.field_update_counter += 1
    if ctx.field_update_counter % FIELD_HSV_UPDATE_INTERVAL != 0:
        return
    h, w = frame.shape[:2]
    roi_hsv = hsv[
        int(h * FIELD_ROI_Y0_FRAC) : int(h * FIELD_ROI_Y1_FRAC),
        int(w * FIELD_ROI_X0_FRAC) : int(w * FIELD_ROI_X1_FRAC),
    ]
    pixels = roi_hsv.reshape(-1, 3)
    if len(pixels) > FIELD_HSV_CALIB_PX_PER_FRAME:
        idx = np.random.choice(len(pixels), FIELD_HSV_CALIB_PX_PER_FRAME, replace=False)
        pixels = pixels[idx]
    new_hsv_bounds = estimate_hsv_from_pixel_arrays([pixels])
    ctx.field_hsv_bounds = _blend_hsv_bounds(
        ctx.field_hsv_bounds, new_hsv_bounds, FIELD_HSV_BLEND_ALPHA
    )
    if ctx.classifier is not None:
        ctx.classifier.field_hsv = ctx.field_hsv_bounds
        ctx.classifier.field_hsv_ready = True
    ctx.field_hsv_updated_this_frame = True


def _maybe_init_field_hsv(
    frame: np.ndarray, ctx: VideoPipelineContext, hsv: np.ndarray
) -> None:
    """Defer HSV auto-calibration until homography valid or enough green in center ROI."""
    if ctx.field_hsv_bounds is not None or ctx.classifier is None:
        return

    reg_ok = ctx.field_reg.registration_valid
    green_frac = center_roi_green_fraction(frame, hsv)
    if reg_ok or green_frac > 0.15:
        # Extract and subsample ROI pixels now; don't store the full frame
        h, w = frame.shape[:2]
        roi_hsv = hsv[
            int(h * FIELD_ROI_Y0_FRAC) : int(h * FIELD_ROI_Y1_FRAC),
            int(w * FIELD_ROI_X0_FRAC) : int(w * FIELD_ROI_X1_FRAC),
        ]
        pixels = roi_hsv.reshape(-1, 3)
        if len(pixels) > FIELD_HSV_CALIB_PX_PER_FRAME:
            idx = np.random.choice(len(pixels), FIELD_HSV_CALIB_PX_PER_FRAME, replace=False)
            pixels = pixels[idx]
        ctx.auto_frames.append(pixels)
    if len(ctx.auto_frames) >= FIELD_HSV_AUTO_FRAMES:
        ctx.field_hsv_bounds = estimate_hsv_from_pixel_arrays(ctx.auto_frames)
        ctx.classifier.field_hsv = ctx.field_hsv_bounds
        ctx.classifier.field_hsv_ready = True
        ctx.auto_frames.clear()
        ctx.hsv_deferred = False


def _update_field_mask(
    frame: np.ndarray, ctx: VideoPipelineContext, hsv: np.ndarray | None = None
) -> np.ndarray | None:
    if hsv is None:
        import cv2 as _cv2
        hsv = _cv2.cvtColor(frame, _cv2.COLOR_BGR2HSV)
    if ctx.classifier is None and not ctx.filter_field:
        return None

    # Skip Hough+RANSAC on stable frames; stride widens once locked
    streak = ctx.field_reg.valid_streak
    stride = FIELD_REG_UPDATE_STRIDE_STABLE if streak >= FIELD_REG_STABLE_STREAK else 1
    if ctx.frame_idx % stride == 0:
        ctx.field_reg.update(frame, hsv=hsv)
    if ctx.field_reg.registration_valid:
        ctx.registration_valid_frames += 1
        play = ctx.field_reg.playable_mask(frame.shape[0], frame.shape[1])
        if play is not None:
            ctx.cached_field_mask = play
            ctx.field_mask_frame = ctx.frame_idx
            if ctx.classifier is not None:
                ctx.classifier.field_mask = play
                ctx.classifier.homography = ctx.field_reg.homography
            return play

    if ctx.classifier is not None:
        ctx.classifier.homography = None

    should_recompute = (
        ctx.cached_field_mask is None
        or (ctx.frame_idx - ctx.field_mask_frame) >= FIELD_MASK_INTERVAL
        or ctx.field_hsv_updated_this_frame
    )

    if should_recompute:
        if ctx.field_hsv_bounds is not None:
            mask = build_field_mask_from_bounds(frame, *ctx.field_hsv_bounds, hsv=hsv)
        elif ctx.classifier is not None or ctx.filter_field:
            mask = build_field_mask(frame, hsv=hsv)
        else:
            return None
        ctx.cached_field_mask = mask
        ctx.field_mask_frame = ctx.frame_idx

    mask = ctx.cached_field_mask
    if ctx.classifier is not None and mask is not None:
        ctx.classifier.field_mask = mask
    return mask


def _attach_pose(frame: np.ndarray, ctx: VideoPipelineContext, detections: list[dict]) -> None:
    if not ctx.use_pose or not detections:
        return
    if ctx.pose_every > 1 and ctx.frame_idx % ctx.pose_every != 0:
        return
    try:
        if ctx._pose_model is None:
            ctx._pose_model = get_pose_model()
        pose_dets = run_pose_on_frame(ctx._pose_model, frame)
        attach_keypoints_to_detections(detections, pose_dets)
    except Exception as e:
        if ctx.frame_idx == 0:
            print(f"[pose] disabled: {e}")


def _cached_team_labels(ctx: VideoPipelineContext, detections: list[dict]) -> dict[int, int]:
    clf = ctx.classifier
    assert clf is not None
    labels: dict[int, int] = {}
    for det in detections:
        tid = det["track_id"]
        if tid in clf.voter.locked_team:
            labels[tid] = clf.voter.locked_team[tid]
        elif tid in ctx.last_team_labels:
            labels[tid] = ctx.last_team_labels[tid]
        elif tid in clf.voter.last_label:
            labels[tid] = clf.voter.last_label[tid]
        else:
            labels[tid] = -1
    return labels


def _reset_tracker(model: YOLO) -> None:
    """Clear BoT-SORT / ByteTrack internal state after a camera cut."""
    try:
        predictor = getattr(model, 'predictor', None)
        if predictor is None:
            return
        for tracker in getattr(predictor, 'trackers', []):
            if hasattr(tracker, 'reset'):
                tracker.reset()
            else:
                for attr in ('tracked_stracks', 'lost_stracks', 'removed_stracks'):
                    if hasattr(tracker, attr):
                        getattr(tracker, attr).clear()
    except Exception:
        pass  # never crash the pipeline during cleanup


def _flow_interpolate(
    last_detections: list[dict],
    last_gray: np.ndarray | None,
    cur_gray: np.ndarray,
) -> list[dict]:
    """Translate last-frame bboxes by sparse optical flow. Falls back to frozen boxes."""
    if not last_detections or last_gray is None:
        return [{**d, "frame_age": 1} for d in last_detections]

    pts = np.array(
        [[(d["bbox"][0] + d["bbox"][2]) / 2, (d["bbox"][1] + d["bbox"][3]) / 2]
         for d in last_detections],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        last_gray, cur_gray, pts, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    )

    result = []
    for i, det in enumerate(last_detections):
        if status is not None and status[i, 0] == 1 and new_pts is not None:
            dx = float(new_pts[i, 0, 0] - pts[i, 0, 0])
            dy = float(new_pts[i, 0, 1] - pts[i, 0, 1])
        else:
            dx, dy = 0.0, 0.0
        x1, y1, x2, y2 = det["bbox"]
        new_box = (int(x1 + dx), int(y1 + dy), int(x2 + dx), int(y2 + dy))
        result.append({**det, "bbox": new_box, "frame_age": 1})
    return result


def process_video_frame(
    frame: np.ndarray, ctx: VideoPipelineContext
) -> tuple[list[dict], dict[int, int] | None]:
    """Run detection (+ optional team labels) for one frame. Mutates ctx."""
    if ctx.frame_idx == 0:
        get_reassociator().reset()
    ctx.field_hsv_updated_this_frame = False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    _maybe_init_field_hsv(frame, ctx, hsv)
    if ctx.field_hsv_bounds is not None:
        _update_field_hsv_rolling(frame, ctx, hsv)

    if ctx.prev_frame is not None:
        mad = float(
            np.mean(
                np.abs(frame.astype(np.float32) - ctx.prev_frame.astype(np.float32))
            )
        )
        if ctx.camera_cut_cooldown > 0:
            ctx.camera_cut_cooldown -= 1
            ctx.camera_cut_streak = 0
        elif mad > CAMERA_CUT_MAD_THRESHOLD:
            ctx.camera_cut_streak += 1
        else:
            ctx.camera_cut_streak = 0

        if (
            ctx.camera_cut_cooldown == 0
            and ctx.camera_cut_streak >= CAMERA_CUT_CONSECUTIVE_FRAMES
        ):
            print(f"[tracker] camera cut detected at frame {ctx.frame_idx} (MAD={mad:.1f})")
            ctx.camera_cut_streak = 0
            ctx.camera_cut_cooldown = POST_CUT_FRAMES
            _reset_tracker(ctx.model)
            get_reassociator().reset()
            ctx.bbox_smoother.clear()
            ctx.hud_band_history.clear()
            if ctx.classifier is not None:
                ctx.classifier.post_cut_frames = POST_CUT_FRAMES
                if ctx.classifier.state != FootballTeamClassifier.STATE_LOCKED:
                    ctx.field_reg.reset()
                    ctx.auto_frames.clear()

    field_mask = _update_field_mask(frame, ctx, hsv)

    hud_band = None
    if ctx.apply_hud_filter:
        raw_top, raw_bottom = _hud_limits(frame.shape, field_mask=field_mask)
        recent = ctx.hud_band_history[-(HUD_BAND_HOLD_FRAMES - 1) :]
        hud_band = hold_hud_band_limits(raw_top, raw_bottom, recent)
        ctx.hud_band_history.append((raw_top, raw_bottom))
        if len(ctx.hud_band_history) > HUD_BAND_HOLD_FRAMES:
            ctx.hud_band_history = ctx.hud_band_history[-HUD_BAND_HOLD_FRAMES:]

    run_yolo = ctx.detect_every <= 1 or ctx.frame_idx % ctx.detect_every == 0
    cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if run_yolo:
        detections = extract_tracked_detections(
            ctx.model,
            frame,
            frame_idx=ctx.frame_idx,
            apply_hud_filter=ctx.apply_hud_filter,
            stats=ctx.det_stats,
            run_inference=True,
            retina_masks=ctx.retina_masks,
            tracker_cfg=ctx.tracker_cfg,
            player_conf=ctx.player_conf,
            player_iou=ctx.player_iou,
            player_imgsz=ctx.player_imgsz,
            player_max_det=ctx.player_max_det,
            player_half=ctx.player_half,
            player_device=ctx.player_device,
            field_mask=field_mask,
            hud_band=hud_band,
        )
        detections = suppress_duplicate_tracks(detections)
        detections = get_reassociator().apply(frame, detections, ctx.frame_idx)
        _attach_pose(frame, ctx, detections)
        detections = [{**d, "frame_age": 0} for d in detections]
        ctx.last_detections = detections
        ctx.last_gray = cur_gray

        cur_ids = {d["track_id"] for d in detections}
        ctx.track_id_changes += len(cur_ids - ctx.prev_track_ids)
        ctx.prev_track_ids = cur_ids
    else:
        detections = _flow_interpolate(ctx.last_detections, ctx.last_gray, cur_gray)

    if ctx.filter_field and detections and field_mask is not None:
        before = len(detections)
        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        confs = np.array([d["conf"] for d in detections])
        tids = np.array([d["track_id"] for d in detections])
        xyxy, confs, tids = filter_by_field_area(
            frame, xyxy, confs, tids, field_mask, min_field_frac=FIELD_FOOT_MIN_FRAC
        )
        keep = {int(t) for t in tids} if len(tids) else set()
        detections = [d for d in detections if d["track_id"] in keep]
        ctx.field_rejected_total += before - len(detections)

    # EMA bbox smoothing: damp jitter while tracking fast-moving players.
    if BBOX_EMA_ALPHA < 1.0 and detections:
        alive = set()
        for det in detections:
            tid = det["track_id"]
            alive.add(tid)
            raw = np.array(det["bbox"], dtype=np.float32)
            if tid in ctx.bbox_smoother:
                smoothed = BBOX_EMA_ALPHA * raw + (1.0 - BBOX_EMA_ALPHA) * ctx.bbox_smoother[tid]
            else:
                smoothed = raw
            ctx.bbox_smoother[tid] = smoothed
            det["bbox"] = tuple(int(v) for v in smoothed)
        # prune stale tracks from smoother
        for tid in list(ctx.bbox_smoother):
            if tid not in alive:
                del ctx.bbox_smoother[tid]

    if ctx.debug_filters and ctx.frame_idx > 0 and ctx.frame_idx % 300 == 0:
        print(f"[filters] frame {ctx.frame_idx}: field_rejected={ctx.field_rejected_total}")

    team_labels = None
    if ctx.classifier is not None:
        ctx.classifier.frame_idx = ctx.frame_idx
        if (
            not run_yolo
            and ctx.classifier.state == FootballTeamClassifier.STATE_LOCKED
            and ctx.classifier.post_cut_frames == 0
        ):
            team_labels = _cached_team_labels(ctx, detections)
        else:
            team_labels = ctx.classifier.process_frame(frame, detections)
            if (
                ctx.classifier.state == FootballTeamClassifier.STATE_LOCKED
                and ctx.classifier.locked_frame_index is None
            ):
                ctx.classifier.locked_frame_index = ctx.frame_idx
            elif (
                ctx.classifier.state == FootballTeamClassifier.STATE_LOCKED
                and ctx.classifier.locked_frame_index == -1
            ):
                ctx.classifier.locked_frame_index = ctx.frame_idx
            ctx.last_team_labels = dict(team_labels)

    ctx.prev_frame = frame.copy()
    ctx.frame_idx += 1
    return detections, team_labels
