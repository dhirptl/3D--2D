"""Render team-colored boxes and classifier state overlay."""

import cv2
import numpy as np

from src.config import QUALITY_FRAMES, WARMUP_MIN_UNIQUE_TRACKS
from src.team_classifier import FootballTeamClassifier

COLOR_GREY = (128, 128, 128)
COLOR_REF = (0, 255, 255)
COLOR_TEAM0 = (0, 0, 255)
COLOR_TEAM1 = (255, 0, 0)
COLOR_PREVIEW0 = (0, 128, 255)
COLOR_PREVIEW1 = (255, 128, 0)


def team_color(label: int, state: str, *, preview: bool = False) -> tuple[int, int, int]:
    if label == -2:
        return COLOR_REF
    if state == FootballTeamClassifier.STATE_LOCKED:
        if label == 0:
            return COLOR_TEAM0
        if label == 1:
            return COLOR_TEAM1
        return COLOR_GREY
    if state == FootballTeamClassifier.STATE_WARMUP and preview and label >= 0:
        return COLOR_PREVIEW0 if label == 0 else COLOR_PREVIEW1
    return COLOR_GREY


def _blend_mask_roi(
    out: np.ndarray,
    crop_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    roi = out[y1:y2, x1:x2]
    if crop_mask.shape[:2] != roi.shape[:2]:
        m = cv2.resize(crop_mask, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    else:
        m = crop_mask > 0
    if not np.any(m):
        return
    color_arr = np.array(color, dtype=np.float32)
    roi[m] = (roi[m].astype(np.float32) * 0.6 + color_arr * 0.4).astype(np.uint8)


def draw_teams(
    frame: np.ndarray,
    detections: list[dict],
    team_labels: dict[int, int],
    classifier: FootballTeamClassifier,
    *,
    show_masks: bool = False,
    debug_teams: bool = False,
) -> np.ndarray:
    out = frame.copy()
    state = classifier.state
    warmup_count = classifier.warmup_count
    unique_tracks = classifier.warmup_unique_tracks
    use_preview = state == FootballTeamClassifier.STATE_WARMUP and classifier.preview_centroids is not None

    flash = classifier.cal_log.tick_flash()
    if flash:
        cv2.putText(
            out, flash[:60], (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA,
        )

    for det in detections:
        tid = det["track_id"]
        x1, y1, x2, y2 = det["bbox"]
        label = team_labels.get(tid, -1)
        if state == FootballTeamClassifier.STATE_LOCKED and label < 0:
            label = classifier.voter.last_label.get(tid, -1)

        color = team_color(label, state, preview=use_preview)
        thickness = 2 if state == FootballTeamClassifier.STATE_LOCKED else 1

        if show_masks and det.get("mask") is not None:
            _blend_mask_roi(out, det["mask"], (x1, y1, x2, y2), color)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        conf = det.get("conf", 0.0)
        tag = f"ID{tid}"
        if label >= 0:
            tag += f" T{label}"
        elif label == -2:
            tag += " REF"
        tag += f" {conf:.2f}"

        if debug_teams:
            dbg = classifier.last_frame_debug.get(tid, {})
            if "dist0" in dbg:
                tag += f" d0={dbg['dist0']:.1f} d1={dbg['dist1']:.1f}"
            elif "reject" in dbg:
                tag += f" [{dbg['reject']}]"

        cv2.putText(
            out, tag, (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    if state == FootballTeamClassifier.STATE_WARMUP:
        status = (
            f"WARMUP {warmup_count}/{QUALITY_FRAMES} "
            f"tracks={unique_tracks}/{WARMUP_MIN_UNIQUE_TRACKS}"
        )
    elif state == FootballTeamClassifier.STATE_CALIBRATING:
        status = "CALIBRATING"
    else:
        status = "LOCKED"
    cv2.putText(
        out, status, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return out
