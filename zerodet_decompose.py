#!/usr/bin/env python3
"""
zerodet_decompose.py  —  Decompose a clip's "zero-detection" frames into the
buckets that actually matter, and render contact sheets for the buckets that
need visual verification (raw=0, raw 1-2, and the HUD-eaten frames).

WHY THIS EXISTS
---------------
`zero_det_rate` is a single number that conflates >=5 different phenomena:
genuinely empty frames, sparse untrackable singletons, formation-boundary
activation delay, HUD/field-band filtering, and real recall misses. That
conflation is what turned a non-bug into a multi-day investigation. This tool
reports the breakdown so the number is never read naively again.

WHAT IT READS
-------------
This script only READS; it never mutates the repo or the model.
  (a) RAW decomposition (faithful, no pipeline internals): runs model.predict
      per frame and bins by raw detection count. Enough to verify the raw=0
      bucket — the one you're about to exclude from the metric.
  (b) PIPELINE decomposition (optional): if you pass --jsonl pointing at the
      file emitted by the permanent ZERODET_DEBUG instrumentation in
      src/detection.py, it merges the real id-path / HUD truth. The HUD/id-path
      split can ONLY come from the live pipeline, because the homography field
      mask is built during tracking and is not reconstructable here.

USAGE
-----
  # raw decomposition + raw=0 verification render
  python zerodet_decompose.py --source 1minclip.mov --conf 0.18 \
      --render raw0 --render-out outputs/raw0_sheet

  # merge live-pipeline truth, then render the HUD-eaten frames
  ZERODET_DEBUG=outputs/zerodet.jsonl python -m src.run_tracker --source 1minclip.mov ...
  python zerodet_decompose.py --source 1minclip.mov --conf 0.18 \
      --jsonl outputs/zerodet.jsonl --render hud --render-out outputs/hud_sheet
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

# --- repo imports (run from repo root so `src` is importable) ---
from src.config import PLAYER_CLASS_ID, PLAYER_IMGSZ, SEG_MODEL_PATH
from ultralytics import YOLO


def raw_counts(source: str, conf: float, max_frames: int | None):
    """Faithful raw decomposition: per-frame raw player-detection count + max conf.

    Uses model.predict (no tracker, no HUD filter) so the result reflects ONLY
    what the model sees — exactly the floor the investigation cared about.
    """
    model = YOLO(str(SEG_MODEL_PATH))
    cap = cv2.VideoCapture(source)
    per_frame = []  # (frame_idx, n_raw, max_conf)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        res = model.predict(frame, conf=conf, classes=[PLAYER_CLASS_ID],
                            imgsz=PLAYER_IMGSZ, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            per_frame.append((idx, 0, 0.0))
        else:
            confs = res.boxes.conf.cpu().numpy()
            per_frame.append((idx, len(confs), float(confs.max())))
        idx += 1
    cap.release()
    return per_frame


def bucket_raw(per_frame):
    b = Counter()
    for _, n, _ in per_frame:
        if n == 0:
            b["raw=0"] += 1
        elif n <= 2:
            b["raw 1-2"] += 1
        elif n <= 4:
            b["raw 3-4"] += 1
        else:
            b["raw>=5"] += 1
    return b


def load_jsonl(jsonl_path):
    """Return {frame_idx: record} from the ZERODET_DEBUG JSONL."""
    by_frame = {}
    for line in Path(jsonl_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        by_frame[rec["frame"]] = rec
    return by_frame


def merge_pipeline(per_frame, by_frame):
    """Merge live-pipeline truth (id-path + HUD) keyed by frame_idx."""
    merged = Counter()
    for idx, n_raw, _ in per_frame:
        rec = by_frame.get(idx)
        if rec is None:
            merged["tracked/ok"] += 1
            continue
        path = rec.get("path", "OK")
        if path == "HUD_ATE_ALL":
            merged["HUD field-band ate all"] += 1
        elif path == "ID_NONE":
            rb = rec.get("raw_boxes", n_raw)
            if rb == 0:
                merged["id=None raw=0 (empty)"] += 1
            elif rb <= 2:
                merged["id=None raw 1-2 (sparse/low-conf)"] += 1
            elif rb <= 4:
                merged["id=None raw 3-4 (boundary)"] += 1
            else:
                merged["id=None raw>=5 (formation boundary)"] += 1
        else:
            merged["tracked/ok"] += 1
    return merged


def render_sheet(source, frame_idxs, out_prefix, conf, cols=4, thumb_w=480):
    """Draw raw boxes on sampled frames and tile into contact sheets for the eye."""
    model = YOLO(str(SEG_MODEL_PATH))
    cap = cv2.VideoCapture(source)
    tiles = []
    want = set(frame_idxs)
    idx = 0
    while want:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            want.discard(idx)
            res = model.predict(frame, conf=conf, classes=[PLAYER_CLASS_ID],
                                imgsz=PLAYER_IMGSZ, verbose=False)[0]
            if res.boxes is not None:
                for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, b)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{c:.2f}", (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            h = int(thumb_w * frame.shape[0] / frame.shape[1])
            t = cv2.resize(frame, (thumb_w, h))
            cv2.putText(t, f"f{idx}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            tiles.append(t)
        idx += 1
    cap.release()
    if not tiles:
        print("no frames rendered"); return
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    out = f"{out_prefix}.png"
    cv2.imwrite(out, sheet)
    print(f"contact sheet -> {out}  ({len(tiles)} frames)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--conf", type=float, default=0.18)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--jsonl", default=None,
                    help="optional: live-pipeline JSONL from ZERODET_DEBUG (required for --render hud)")
    ap.add_argument("--render", choices=["raw0", "raw12", "hud"], default=None)
    ap.add_argument("--render-n", type=int, default=12)
    ap.add_argument("--render-out", default="outputs/zerodet_sheet")
    args = ap.parse_args()

    per_frame = raw_counts(args.source, args.conf, args.max_frames)
    total = len(per_frame)
    raw_b = bucket_raw(per_frame)

    print(f"\n=== RAW decomposition  (conf={args.conf}, {total} frames) ===")
    for k in ("raw=0", "raw 1-2", "raw 3-4", "raw>=5"):
        n = raw_b.get(k, 0)
        print(f"  {k:10s} {n:5d}  {100*n/total:5.1f}%")
    raw0 = raw_b.get("raw=0", 0)
    print(f"\n  raw=0 is the bucket you plan to EXCLUDE from zero_det_rate "
          f"({100*raw0/total:.1f}%). Render it (--render raw0) and confirm it's "
          f"non-play before excluding.")

    by_frame = load_jsonl(args.jsonl) if args.jsonl else {}
    if args.jsonl:
        merged = merge_pipeline(per_frame, by_frame)
        print(f"\n=== PIPELINE decomposition (from {args.jsonl}) ===")
        for k, n in merged.most_common():
            print(f"  {k:36s} {n:5d}  {100*n/total:5.1f}%")

    if args.render:
        if args.render == "raw0":
            idxs = [i for i, n, _ in per_frame if n == 0]
            label = "raw=0 (verify these are truly non-play)"
        elif args.render == "raw12":
            idxs = [i for i, n, _ in per_frame if 1 <= n <= 2]
            label = "raw 1-2 (real players vs FP)"
        else:  # hud
            if not by_frame:
                ap.error("--render hud requires --jsonl (HUD frame indices come from the live pipeline)")
            idxs = sorted(f for f, rec in by_frame.items() if rec.get("path") == "HUD_ATE_ALL")
            label = "HUD_ATE_ALL (does the field-band eat real endzone/sideline players?)"
        if not idxs:
            print(f"\nno frames in bucket '{args.render}' to render")
            return
        step = max(1, len(idxs) // args.render_n)
        sample = idxs[::step][:args.render_n]
        print(f"\nrendering {len(sample)} of {len(idxs)} frames: {label}")
        render_sheet(args.source, sample, args.render_out, args.conf)


if __name__ == "__main__":
    main()
