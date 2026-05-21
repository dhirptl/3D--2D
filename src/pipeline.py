"""Shared per-frame video processing for run_tracker and eval_clip."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import (
    CAMERA_CUT_MAD_THRESHOLD,
    FIELD_HSV_AUTO_FRAMES,
    FIELD_HSV_BLEND_ALPHA,
    FIELD_HSV_UPDATE_INTERVAL,
    FIELD_MASK_INTERVAL,
    FIELD_REG_VALID_STREAK,
    POSE_EVERY_DEFAULT,
    POST_CUT_FRAMES,
)
from src.detection import DetectionStats, extract_tracked_detections
from src.field_hsv import build_field_mask_from_bounds, estimate_field_hsv
from src.field_registration import FieldRegistration, center_roi_green_fraction
from src.pose_estimation import attach_keypoints_to_detections, get_pose_model, run_pose_on_frame
from src.post_process import build_field_mask, filter_by_field_area
from src.team_classifier import FootballTeamClassifier


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


def _blend_hsv_bounds(
    old: tuple[int, int, int, int], new: tuple[int, int, int, int], alpha: float
) -> tuple[int, int, int, int]:
    return tuple(int((1 - alpha) * o + alpha * n) for o, n in zip(old, new))


def _update_field_hsv_rolling(frame: np.ndarray, ctx: VideoPipelineContext) -> None:
    if ctx.field_hsv_bounds is None:
        return
    ctx.field_update_counter += 1
    if ctx.field_update_counter % FIELD_HSV_UPDATE_INTERVAL != 0:
        return
    h, w = frame.shape[:2]
    center_strip = frame[h // 3 : 2 * h // 3, w // 3 : 2 * w // 3]
    new_hsv = estimate_field_hsv([center_strip])
    ctx.field_hsv_bounds = _blend_hsv_bounds(
        ctx.field_hsv_bounds, new_hsv, FIELD_HSV_BLEND_ALPHA
    )
    if ctx.classifier is not None:
        ctx.classifier.field_hsv = ctx.field_hsv_bounds
        ctx.classifier.field_hsv_ready = True
    ctx.field_hsv_updated_this_frame = True


def _maybe_init_field_hsv(frame: np.ndarray, ctx: VideoPipelineContext) -> None:
    """Defer HSV auto-calibration until homography valid or enough green in center ROI."""
    if ctx.field_hsv_bounds is not None or ctx.classifier is None:
        return

    reg_ok = ctx.field_reg.registration_valid
    green_frac = center_roi_green_fraction(frame)
    if reg_ok or green_frac > 0.15:
        ctx.auto_frames.append(frame)
    if len(ctx.auto_frames) >= FIELD_HSV_AUTO_FRAMES:
        ctx.field_hsv_bounds = estimate_field_hsv(ctx.auto_frames)
        ctx.classifier.field_hsv = ctx.field_hsv_bounds
        ctx.classifier.field_hsv_ready = True
        ctx.auto_frames.clear()
        ctx.hsv_deferred = False


def _update_field_mask(frame: np.ndarray, ctx: VideoPipelineContext) -> np.ndarray | None:
    if ctx.classifier is None and not ctx.filter_field:
        return None

    ctx.field_reg.update(frame)
    if ctx.field_reg.registration_valid:
        ctx.registration_valid_frames += 1
        play = ctx.field_reg.playable_mask(frame.shape[0], frame.shape[1])
        if play is not None:
            ctx.cached_field_mask = play
            ctx.field_mask_frame = ctx.frame_idx
            if ctx.classifier is not None:
                ctx.classifier.field_mask = play
            return play

    should_recompute = (
        ctx.cached_field_mask is None
        or (ctx.frame_idx - ctx.field_mask_frame) >= FIELD_MASK_INTERVAL
        or ctx.field_hsv_updated_this_frame
    )

    if should_recompute:
        if ctx.field_hsv_bounds is not None:
            mask = build_field_mask_from_bounds(frame, *ctx.field_hsv_bounds)
        elif ctx.classifier is not None or ctx.filter_field:
            mask = build_field_mask(frame)
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


def process_video_frame(
    frame: np.ndarray, ctx: VideoPipelineContext
) -> tuple[list[dict], dict[int, int] | None]:
    """Run detection (+ optional team labels) for one frame. Mutates ctx."""
    ctx.field_hsv_updated_this_frame = False

    _maybe_init_field_hsv(frame, ctx)
    if ctx.field_hsv_bounds is not None:
        _update_field_hsv_rolling(frame, ctx)

    if ctx.prev_frame is not None and ctx.classifier is not None:
        mad = float(
            np.mean(
                np.abs(frame.astype(np.float32) - ctx.prev_frame.astype(np.float32))
            )
        )
        if mad > CAMERA_CUT_MAD_THRESHOLD:
            print(f"[tracker] camera cut detected at frame {ctx.frame_idx}")
            ctx.classifier.post_cut_frames = POST_CUT_FRAMES

    field_mask = _update_field_mask(frame, ctx)

    run_yolo = ctx.detect_every <= 1 or ctx.frame_idx % ctx.detect_every == 0
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
        )
        _attach_pose(frame, ctx, detections)
        detections = [{**d, "frame_age": 0} for d in detections]
        ctx.last_detections = detections

        cur_ids = {d["track_id"] for d in detections}
        ctx.track_id_changes += len(cur_ids - ctx.prev_track_ids)
        ctx.prev_track_ids = cur_ids
    else:
        detections = [{**d, "frame_age": 1} for d in ctx.last_detections]

    if ctx.filter_field and detections and field_mask is not None:
        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        confs = np.array([d["conf"] for d in detections])
        tids = np.array([d["track_id"] for d in detections])
        xyxy, confs, tids = filter_by_field_area(frame, xyxy, confs, tids, field_mask)
        keep = {int(t) for t in tids} if len(tids) else set()
        detections = [d for d in detections if d["track_id"] in keep]

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
