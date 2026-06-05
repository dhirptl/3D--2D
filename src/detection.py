"""Extract tracked detections with bbox-crop masks from YOLO-Seg + tracker."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import STrack as _STrack

# ByteTrack only sets is_activated=True for frame_id==1; all later new tracks start tentative
# and are removed if missed the very next frame, preventing any track from being confirmed.
# Patch to confirm immediately so tracks enter Lost (not Removed) state when missed,
# allowing re_activate() to reclaim the same ID on re-detection.
_orig_strack_activate = _STrack.activate
def _activate_immediate(self, kalman_filter, frame_id):
    _orig_strack_activate(self, kalman_filter, frame_id)
    self.is_activated = True

_STrack.activate = _activate_immediate

from src.config import (
    PLAYER_CLASS_ID,
    PLAYER_DEVICE,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_HALF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    TRACKER_CFG,
)
from src.detection_ids import track_ids_from_boxes
from src.mask_utils import (
    bbox_fill_mask,
    crop_mask_to_bbox,
    get_instance_mask_full,
    mask_area,
)
from src.post_process import _hud_limits, filter_hud_detections

logger = logging.getLogger(__name__)


def resolve_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Zero-detection decomposition: env-gated, zero-cost in production. Set ZERODET_DEBUG to a
# path to log which path produced each zero-det frame (ID_NONE vs HUD_ATE_ALL), then feed the
# JSONL to zerodet_decompose.py. See plan: zero_det_rate conflates >=5 distinct phenomena.
_ZERODET_LOG = os.environ.get("ZERODET_DEBUG")  # path or None
if _ZERODET_LOG:
    open(_ZERODET_LOG, "w").close()  # truncate once at import so runs don't accumulate stale frames

def _zerodet_emit(rec: dict) -> None:
    if _ZERODET_LOG:
        with open(_ZERODET_LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")


@dataclass
class DetectionStats:
    frames: int = 0
    total_dets: int = 0
    zero_det_frames: int = 0
    missing_mask_warnings: int = 0
    max_dets: int = 0
    verbose: bool = False
    _per_frame_counts: list | None = None

    def record(self, n: int) -> None:
        self.frames += 1
        self.total_dets += n
        self.max_dets = max(self.max_dets, n)
        if self.verbose:
            if self._per_frame_counts is None:
                self._per_frame_counts = []
            self._per_frame_counts.append(n)
        if n == 0:
            self.zero_det_frames += 1

    def summary(self) -> str:
        avg = self.total_dets / self.frames if self.frames else 0
        return (
            f"Detection summary: avg={avg:.1f}/frame max={self.max_dets} "
            f"zero_frames={self.zero_det_frames} mask_warnings={self.missing_mask_warnings}"
        )


def _build_detections_from_arrays(
    frame: np.ndarray,
    result,
    *,
    frame_idx: int,
    xyxy: np.ndarray,
    confs: np.ndarray,
    track_ids: np.ndarray,
    mask_indices: np.ndarray,
    result_masks,
    masks_missing: bool,
    stats: DetectionStats | None,
) -> list[dict]:
    h, w = frame.shape[:2]
    detections = []
    for i, k in enumerate(mask_indices):
        box = tuple(map(int, xyxy[i]))
        if masks_missing:
            crop_mask = bbox_fill_mask(box)
        else:
            full_mask = get_instance_mask_full(result_masks, int(k), (h, w))
            if full_mask is None:
                if stats:
                    stats.missing_mask_warnings += 1
                crop_mask = bbox_fill_mask(box)
            else:
                crop_mask = crop_mask_to_bbox(full_mask, box)
        if mask_area(crop_mask) == 0:
            if stats:
                stats.missing_mask_warnings += 1
            continue
        detections.append({
            "track_id": int(track_ids[i]),
            "bbox": box,
            "conf": float(confs[i]),
            "mask": crop_mask,
            "mask_fallback": masks_missing,
        })
    return detections


def extract_tracked_detections(
    model: YOLO,
    frame: np.ndarray,
    *,
    frame_idx: int = 0,
    apply_hud_filter: bool = True,
    stats: DetectionStats | None = None,
    run_inference: bool = True,
    retina_masks: bool = False,
    tracker_cfg: Path | None = None,
    player_conf: float | None = None,
    player_iou: float | None = None,
    player_imgsz: int | None = None,
    player_max_det: int | None = None,
    player_half: bool | None = None,
    player_device: str | None = None,
    field_mask: np.ndarray | None = None,
    hud_band: tuple[float, float] | None = None,
    hud_fail_open: bool = True,
) -> list[dict]:
    if not run_inference:
        return []

    cfg = tracker_cfg or TRACKER_CFG
    tracker = str(cfg) if cfg.exists() else "bytetrack.yaml"
    conf = PLAYER_PREDICT_CONF if player_conf is None else player_conf
    iou = PLAYER_PREDICT_IOU if player_iou is None else player_iou
    imgsz = PLAYER_IMGSZ if player_imgsz is None else player_imgsz
    max_det = PLAYER_PREDICT_MAX_DET if player_max_det is None else player_max_det
    half = PLAYER_PREDICT_HALF if player_half is None else player_half
    device = resolve_device(PLAYER_DEVICE if player_device is None else player_device)
    try:
        results = model.track(
            frame,
            classes=[PLAYER_CLASS_ID],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            tracker=tracker,
            persist=True,
            retina_masks=retina_masks,
            half=half,
            device=device,
            verbose=False,
        )
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in ("shape", "size", "mismatch", "broadcast")):
            logger.warning("frame %d: tracker state error (%s); skipping frame", frame_idx, e)
            if stats:
                stats.missing_mask_warnings += 1
            return []
        raise
    result = results[0]

    if result.boxes is None:
        _zerodet_emit({"frame": frame_idx, "path": "ID_NONE", "raw_boxes": 0})
        if stats:
            stats.record(0)
        return []

    orig_xyxy = result.boxes.xyxy.cpu().numpy()
    if orig_xyxy.shape[0] == 0:
        _zerodet_emit({"frame": frame_idx, "path": "ID_NONE", "raw_boxes": 0})
        if stats:
            stats.record(0)
        return []

    n_raw = int(orig_xyxy.shape[0])
    if result.boxes.id is None:
        _zerodet_emit({"frame": frame_idx, "path": "ID_NONE", "raw_boxes": n_raw})

    confs = result.boxes.conf.cpu().numpy()
    track_ids = track_ids_from_boxes(result.boxes)
    result_masks = result.masks
    masks_missing = result_masks is None

    if masks_missing:
        if stats:
            stats.missing_mask_warnings += 1
        logger.warning("frame %d: boxes present but masks is None; using bbox fallback", frame_idx)

    xyxy = orig_xyxy
    hud_idx = np.arange(len(orig_xyxy))
    if apply_hud_filter:
        n_before_hud = len(xyxy)
        pre_hud_xyxy = orig_xyxy.copy()
        pre_confs = confs.copy()
        pre_tids = track_ids.copy()
        pre_idx = hud_idx.copy()
        xyxy, confs, track_ids, hud_idx = filter_hud_detections(
            frame.shape,
            xyxy,
            confs,
            track_ids,
            field_mask=field_mask,
            hud_band=hud_band,
            fail_open=hud_fail_open,
        )
        if len(xyxy) == 0 and n_before_hud > 0 and hud_fail_open:
            xyxy = pre_hud_xyxy
            confs = pre_confs
            track_ids = pre_tids
            hud_idx = pre_idx
        elif len(xyxy) == 0 and n_before_hud > 0:
            fm = "none" if field_mask is None else (
                "empty" if not field_mask.any() else "valid"
            )
            rec = {
                "frame": frame_idx,
                "path": "HUD_ATE_ALL",
                "raw_boxes": int(n_before_hud),
                "field_mask": fm,
            }
            if _ZERODET_LOG:
                if hud_band is not None:
                    top_lim, bot_lim = hud_band
                else:
                    top_lim, bot_lim = _hud_limits(frame.shape, field_mask=field_mask)
                rec["frame_h"] = int(frame.shape[0])
                rec["band"] = [round(float(top_lim), 1), round(float(bot_lim), 1)]
                rec["box_cy"] = [
                    round(float((b[1] + b[3]) / 2), 1) for b in pre_hud_xyxy
                ]
            _zerodet_emit(rec)
            if stats:
                stats.record(0)
            return []

    detections = _build_detections_from_arrays(
        frame,
        result,
        frame_idx=frame_idx,
        xyxy=xyxy,
        confs=confs,
        track_ids=track_ids,
        mask_indices=hud_idx,
        result_masks=result_masks,
        masks_missing=masks_missing,
        stats=stats,
    )

    if stats:
        stats.record(len(detections))
    return detections
