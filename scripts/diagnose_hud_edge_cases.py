#!/usr/bin/env python3
"""Classify HUD_ATE_ALL frames and simulate counterfactual rescues."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    HUD_FIELD_BOTTOM_MARGIN_PX,
    HUD_FIELD_TOP_MARGIN_PX,
    PLAYER_CLASS_ID,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    SEG_MODEL_PATH,
)
from src.pipeline import VideoPipelineContext, _update_field_mask
from src.post_process import _hud_limits
from src.team_classifier import FootballTeamClassifier
from src.zerodet_metrics import load_zerodet_jsonl

OUT_JSON = ROOT / "outputs" / "hud_edge_diagnosis.json"
OUT_MD = ROOT / "outputs" / "hud_edge_diagnosis.md"
OUT_SHEET = ROOT / "outputs" / "hud_rescue_sheet.png"

NARROW_BAND_FRAC = 0.15
TRANSIENT_NEIGHBOR_DETS = 10
TRANSIENT_WINDOW = 2
FOOT_FRAC_THRESHOLDS = (0.05, 0.08, 0.10, 0.15, 0.20)
MARGIN_SWEEPS = (0, 10, 20, 40)
TEMPORAL_WINDOW = 5


def foot_field_frac(
    box: np.ndarray,
    mask: np.ndarray,
    fh: int,
    fw: int,
) -> float:
    x1, y1, x2, y2 = map(int, box)
    box_h = max(1, y2 - y1)
    bottom_y1 = int(y2 - (box_h * 0.15))
    foot_region = mask[
        max(0, bottom_y1) : min(fh, y2),
        max(0, x1) : min(fw, x2),
    ]
    if foot_region.size == 0:
        return 0.0
    return float(np.count_nonzero(foot_region) / foot_region.size)


def _hud_limits_with_margins(
    frame_shape,
    field_mask: np.ndarray | None,
    *,
    top_extra: int = 0,
    bottom_extra: int = 0,
) -> tuple[float, float]:
    h = frame_shape[0]
    if field_mask is not None and field_mask.size:
        rows = np.where(field_mask.any(axis=1))[0]
        if len(rows):
            top = max(0, int(rows[0]) - HUD_FIELD_TOP_MARGIN_PX - top_extra)
            bot = min(h, int(rows[-1]) + HUD_FIELD_BOTTOM_MARGIN_PX + bottom_extra)
            if bot > top:
                return float(top), float(bot)
    from src.config import HUD_BOTTOM_PCT, HUD_TOP_PCT

    return h * HUD_TOP_PCT, h - (h * HUD_BOTTOM_PCT)


def _cy_keep_mask(xyxy: np.ndarray, top: float, bottom: float) -> np.ndarray:
    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2
    return (cy > top) & (cy < bottom)


def _foot_rescue_mask(
    xyxy: np.ndarray,
    field_mask: np.ndarray | None,
    fh: int,
    fw: int,
    min_frac: float,
) -> np.ndarray:
    if field_mask is None or not field_mask.any():
        return np.zeros(len(xyxy), dtype=bool)
    return np.array(
        [foot_field_frac(box, field_mask, fh, fw) >= min_frac for box in xyxy],
        dtype=bool,
    )


def _predict_boxes(model: YOLO, frame: np.ndarray, conf: float) -> np.ndarray:
    res = model.predict(
        frame,
        conf=conf,
        classes=[PLAYER_CLASS_ID],
        imgsz=PLAYER_IMGSZ,
        verbose=False,
    )[0]
    if res.boxes is None or len(res.boxes) == 0:
        return np.empty((0, 4), dtype=float)
    return res.boxes.xyxy.cpu().numpy()


@dataclass
class FrameReplay:
    field_mask: np.ndarray | None
    top_limit: float
    bottom_limit: float


def replay_field_masks(source: Path, hud_frames: set[int]) -> dict[int, FrameReplay]:
    """Walk clip once; mirror pipeline field-mask path for HUD frames."""
    ctx = VideoPipelineContext(
        model=YOLO(str(SEG_MODEL_PATH)),
        classifier=FootballTeamClassifier(),
        filter_field=False,
    )
    cap = cv2.VideoCapture(str(source))
    replay: dict[int, FrameReplay] = {}
    fi = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ctx.frame_idx = fi
        mask = _update_field_mask(frame, ctx, hsv)
        if fi in hud_frames:
            top, bot = _hud_limits(frame.shape, field_mask=mask)
            replay[fi] = FrameReplay(field_mask=mask, top_limit=top, bottom_limit=bot)
        fi += 1
    cap.release()
    return replay


def replay_det_counts(
    model: YOLO,
    source: Path,
    frame_idxs: set[int],
    conf: float,
) -> dict[int, int]:
    if not frame_idxs:
        return {}
    cap = cv2.VideoCapture(str(source))
    counts: dict[int, int] = {}
    fi = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if fi in frame_idxs:
            counts[fi] = len(_predict_boxes(model, frame, conf))
        fi += 1
    cap.release()
    return counts


def classify_primary(
    *,
    raw_boxes: int,
    center_eaten: bool,
    transient: bool,
    narrow_band: bool,
) -> str:
    if raw_boxes <= 2:
        return "SparseRaw"
    if center_eaten:
        return "CenterEaten"
    if transient:
        return "Transient"
    if narrow_band:
        return "NarrowBand"
    return "LegitHUD"


def render_sheet(
    source: Path,
    model: YOLO,
    frames: list[dict],
    conf: float,
    out_path: Path,
    *,
    cols: int = 4,
    thumb_w: int = 480,
) -> None:
    if not frames:
        return
    cap = cv2.VideoCapture(str(source))
    want = {f["frame"]: f for f in frames}
    tiles = []
    fi = 0
    while cap.isOpened() and want:
        ok, frame = cap.read()
        if not ok:
            break
        if fi in want:
            rec = want.pop(fi)
            top, bot = rec["band"]
            cv2.line(frame, (0, int(top)), (frame.shape[1], int(top)), (0, 0, 255), 2)
            cv2.line(frame, (0, int(bot)), (frame.shape[1], int(bot)), (0, 0, 255), 2)
            boxes = _predict_boxes(model, frame, conf)
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) / 2
                color = (0, 255, 255)
                if rec.get("foot_rescue") and rec.get("field_mask") is not None:
                    ff = foot_field_frac(box, rec["field_mask"], frame.shape[0], frame.shape[1])
                    if ff >= rec.get("best_foot_thresh", 0.10):
                        color = (0, 255, 0)
                elif top < cy < bot:
                    color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            h = int(thumb_w * frame.shape[0] / frame.shape[1])
            tile = cv2.resize(frame, (thumb_w, h))
            cv2.putText(
                tile,
                f"f{fi} {rec['primary']}",
                (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )
            tiles.append(tile)
        fi += 1
    cap.release()
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(out_path), np.vstack(rows))
    print(f"contact sheet -> {out_path} ({len(tiles)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(ROOT / "1minclip.mov"))
    ap.add_argument("--jsonl", default=str(ROOT / "outputs" / "zerodet_post.jsonl"))
    ap.add_argument("--conf", type=float, default=PLAYER_PREDICT_CONF)
    ap.add_argument("--total-frames", type=int, default=None)
    args = ap.parse_args()

    source = Path(args.source)
    jsonl = Path(args.jsonl)
    by_frame = load_zerodet_jsonl(jsonl)
    hud_recs = {
        f: rec for f, rec in by_frame.items() if rec.get("path") == "HUD_ATE_ALL"
    }
    if not hud_recs:
        raise SystemExit(f"No HUD_ATE_ALL records in {jsonl}")

    cap = cv2.VideoCapture(str(source))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if args.total_frames is None else args.total_frames
    cap.release()

    hud_frames = set(hud_recs)
    print(f"Replaying field masks for {len(hud_frames)} HUD frames...")
    replay = replay_field_masks(source, hud_frames)

    neighbor_idxs: set[int] = set()
    for fi in hud_frames:
        for d in range(-TRANSIENT_WINDOW, TRANSIENT_WINDOW + 1):
            n = fi + d
            if 0 <= n < total:
                neighbor_idxs.add(n)

    model = YOLO(str(SEG_MODEL_PATH))
    print(f"Predicting neighbor det counts ({len(neighbor_idxs)} frames)...")
    det_counts = replay_det_counts(model, source, neighbor_idxs, args.conf)

    # band history for temporal counterfactual (full clip replay of limits)
    print("Building band history for temporal counterfactual...")
    band_history: dict[int, tuple[float, float]] = {fi: (r.top_limit, r.bottom_limit) for fi, r in replay.items()}
    # extend with all frames via lightweight second pass
    ctx = VideoPipelineContext(
        model=model,
        classifier=FootballTeamClassifier(),
        filter_field=False,
    )
    cap = cv2.VideoCapture(str(source))
    fi = 0
    all_bands: dict[int, tuple[float, float]] = {}
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ctx.frame_idx = fi
        mask = _update_field_mask(frame, ctx, hsv)
        all_bands[fi] = _hud_limits(frame.shape, field_mask=mask)
        fi += 1
    cap.release()
    total = fi

    per_frame: list[dict] = []
    foot_rescue_by_thresh: Counter = Counter()
    center_eaten_by_thresh: Counter = Counter()
    margin_rescue: Counter = Counter()
    temporal_rescue = 0
    primary_counts: Counter = Counter()

    for frame_idx, rec in sorted(hud_recs.items()):
        rep = replay.get(frame_idx)
        band = rec.get("band") or (
            [rep.top_limit, rep.bottom_limit] if rep else [0, 0]
        )
        top, bot = float(band[0]), float(band[1])
        fh = int(rec.get("frame_h", 1080))
        span_frac = (bot - top) / fh if fh else 0.0
        raw_boxes = int(rec.get("raw_boxes", 0))

        cap = cv2.VideoCapture(str(source))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        xyxy = _predict_boxes(model, frame, args.conf)
        mask = rep.field_mask if rep else None

        cy_keep = _cy_keep_mask(xyxy, top, bot) if len(xyxy) else np.array([], dtype=bool)
        best_foot_thresh = None
        foot_rescue = False
        for thr in FOOT_FRAC_THRESHOLDS:
            if np.any(_foot_rescue_mask(xyxy, mask, fh, frame.shape[1], thr)):
                foot_rescue_by_thresh[thr] += 1
        for thr in FOOT_FRAC_THRESHOLDS:
            if np.any(_foot_rescue_mask(xyxy, mask, fh, frame.shape[1], thr)):
                foot_rescue = True
                best_foot_thresh = thr
                break

        center_eaten = foot_rescue and not np.any(cy_keep)
        narrow_band = span_frac < NARROW_BAND_FRAC
        neighbors_ok = all(
            det_counts.get(frame_idx + d, 0) >= TRANSIENT_NEIGHBOR_DETS
            for d in (-TRANSIENT_WINDOW, TRANSIENT_WINDOW)
            if 0 <= frame_idx + d < total
        )
        transient = neighbors_ok and raw_boxes > 2

        primary = classify_primary(
            raw_boxes=raw_boxes,
            center_eaten=center_eaten,
            transient=transient,
            narrow_band=narrow_band,
        )
        primary_counts[primary] += 1
        if primary == "CenterEaten" and best_foot_thresh is not None:
            center_eaten_by_thresh[best_foot_thresh] += 1

        margin_ok = None
        for extra in MARGIN_SWEEPS:
            if extra == 0:
                continue
            t2, b2 = _hud_limits_with_margins(
                frame.shape, mask, top_extra=extra, bottom_extra=extra
            )
            if np.any(_cy_keep_mask(xyxy, t2, b2)):
                margin_rescue[extra] += 1
                if margin_ok is None:
                    margin_ok = extra

        # temporal: widen band using prior frames
        hist_tops = []
        hist_bots = []
        for j in range(max(0, frame_idx - TEMPORAL_WINDOW + 1), frame_idx + 1):
            if j in all_bands:
                hist_tops.append(all_bands[j][0])
                hist_bots.append(all_bands[j][1])
        temporal_ok = False
        if hist_tops:
            t_hold = min(hist_tops)
            b_hold = max(hist_bots)
            temporal_ok = bool(np.any(_cy_keep_mask(xyxy, t_hold, b_hold)))
            if transient and temporal_ok:
                temporal_rescue += 1

        per_frame.append(
            {
                "frame": frame_idx,
                "raw_boxes": raw_boxes,
                "band": [top, bot],
                "band_span_frac": round(span_frac, 4),
                "field_mask_status": rec.get("field_mask", "?"),
                "primary": primary,
                "center_eaten": center_eaten,
                "foot_rescue": foot_rescue,
                "best_foot_thresh": best_foot_thresh,
                "narrow_band": narrow_band,
                "transient": transient,
                "margin_rescue_px": margin_ok,
                "temporal_rescue": temporal_ok,
            }
        )

    n_hud = len(per_frame)
    foot_rescue_total = sum(1 for e in per_frame if e["foot_rescue"])
    center_eaten_n = primary_counts["CenterEaten"]

    best_thr = max(FOOT_FRAC_THRESHOLDS, key=lambda t: center_eaten_by_thresh.get(t, 0))
    if not center_eaten_by_thresh:
        best_thr = 0.10

    summary = {
        "source": str(source),
        "jsonl": str(jsonl),
        "total_frames": total,
        "hud_ate_all_frames": n_hud,
        "baseline_actionable_hud": n_hud,
        "primary_buckets": dict(primary_counts),
        "foot_rescue_any": foot_rescue_total,
        "foot_rescue_by_threshold": dict(foot_rescue_by_thresh),
        "recommended_foot_thresh": best_thr,
        "margin_rescue_counts": {str(k): v for k, v in margin_rescue.items()},
        "temporal_rescue_transient": temporal_rescue,
        "projected_hud_after_foot_rescue": n_hud - foot_rescue_total,
        "projected_actionable_rate_after_foot": round(
            (n_hud - foot_rescue_total) / max(total, 1), 4
        ),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps({"summary": summary, "frames": per_frame}, indent=2) + "\n"
    )

    lines = [
        "# HUD edge-case diagnosis",
        "",
        f"Source: `{jsonl.name}` ({n_hud} HUD_ATE_ALL frames, {total} total)",
        "",
        "## Primary sub-buckets",
        "",
    ]
    for k, v in primary_counts.most_common():
        lines.append(f"- **{k}**: {v} ({100 * v / n_hud:.1f}%)")
    lines.extend(
        [
            "",
            "## Counterfactual rescues",
            "",
            f"- Foot rescue (any threshold): **{foot_rescue_total}** frames",
            f"- Foot rescue by threshold: {dict(foot_rescue_by_thresh)}",
            f"- Recommended foot threshold: **{best_thr}**",
            f"- Margin sweep recoveries (cy-only): {dict(margin_rescue)}",
            f"- Temporal hold (transient subset): **{temporal_rescue}**",
            "",
            "## Projected impact (foot rescue @ any qualifying threshold)",
            "",
            f"- HUD_ATE_ALL: {n_hud} → **{n_hud - foot_rescue_total}**",
            f"- actionable_zero_det_rate contribution from HUD: "
            f"{100 * n_hud / total:.1f}% → **{100 * (n_hud - foot_rescue_total) / total:.1f}%**",
            "",
            "## Recommended fix",
            "",
        ]
    )
    if center_eaten_n >= primary_counts["NarrowBand"] and center_eaten_n >= primary_counts["Transient"]:
        lines.append(
            f"**Fix A (foot-on-field rescue)** — CenterEaten dominates ({center_eaten_n} frames). "
            f"Use min foot overlap **{best_thr}**."
        )
    elif primary_counts["NarrowBand"] > center_eaten_n:
        lines.append("**Fix B (narrow-band margin scaling)** — NarrowBand dominates.")
    elif primary_counts["Transient"] > center_eaten_n:
        lines.append("**Fix C (temporal band hold)** — Transient dominates.")
    else:
        lines.append("**Fix A (foot-on-field rescue)** — default per hypothesis.")

    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD.read_text())

    center_frames = [e for e in per_frame if e["primary"] == "CenterEaten"][:12]
    if center_frames:
        for e in center_frames:
            rep = replay.get(e["frame"])
            e["field_mask"] = rep.field_mask if rep else None
        render_sheet(source, model, center_frames, args.conf, OUT_SHEET)


if __name__ == "__main__":
    main()
