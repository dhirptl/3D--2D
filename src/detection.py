"""Extract tracked detections with bbox-crop masks from YOLO-Seg + tracker."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import (
    PLAYER_CLASS_ID,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_HALF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    TRACKER_CFG,
)
from src.mask_utils import (
    bbox_fill_mask,
    crop_mask_to_bbox,
    get_instance_mask_full,
    mask_area,
    match_indices,
)
from src.post_process import _hud_limits, filter_hud_detections

logger = logging.getLogger(__name__)

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
    field_mask: np.ndarray | None = None,
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

    if result.boxes is None or result.boxes.id is None:
        n_raw = 0 if result.boxes is None else len(result.boxes)
        _zerodet_emit({"frame": frame_idx, "path": "ID_NONE", "raw_boxes": n_raw})
        if stats:
            stats.record(0)
        return []

    h, w = frame.shape[:2]
    orig_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    result_masks = result.masks
    masks_missing = result_masks is None

    if masks_missing:
        if stats:
            stats.missing_mask_warnings += 1
        logger.warning("frame %d: boxes present but masks is None; using bbox fallback", frame_idx)

    xyxy = orig_xyxy
    if apply_hud_filter:
        n_before_hud = len(xyxy)
        pre_hud_xyxy = orig_xyxy
        xyxy, confs, track_ids = filter_hud_detections(
            frame.shape, xyxy, confs, track_ids, field_mask=field_mask
        )
        if len(xyxy) == 0:
            fm = "none" if field_mask is None else ("empty" if not field_mask.any() else "valid")
            rec = {"frame": frame_idx, "path": "HUD_ATE_ALL",
                   "raw_boxes": int(n_before_hud), "field_mask": fm}
            if _ZERODET_LOG:  # only compute band diagnostics when logging
                top_lim, bot_lim = _hud_limits(frame.shape, field_mask=field_mask)
                rec["frame_h"] = int(frame.shape[0])
                rec["band"] = [round(float(top_lim), 1), round(float(bot_lim), 1)]
                rec["box_cy"] = [round(float((b[1] + b[3]) / 2), 1) for b in pre_hud_xyxy]
            _zerodet_emit(rec)
            if stats:
                stats.record(0)
            return []

    keep_idx = match_indices(xyxy, orig_xyxy) if not masks_missing else list(range(len(xyxy)))

    detections = []
    for i, k in enumerate(keep_idx):
        box = tuple(map(int, xyxy[i]))
        if masks_missing:
            crop_mask = bbox_fill_mask(box)
        else:
            full_mask = get_instance_mask_full(result_masks, k, (h, w))
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

    if stats:
        stats.record(len(detections))
    return detections
