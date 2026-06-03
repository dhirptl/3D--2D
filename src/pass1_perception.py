"""Pass 1: perception-only IR writer (no team decisions)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from ultralytics import YOLO

from src.color_features import build_raw_6d_vector
from src.config import (
    CAMERA_CUT_CONSECUTIVE_FRAMES,
    CAMERA_CUT_MAD_THRESHOLD,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    POSE_EVERY_DEFAULT,
    ROOT,
    SEG_MODEL_PATH,
)
from src.detection import DetectionStats, extract_tracked_detections
from src.helmet_verify import detect_helmets_roi
from src.jersey_ocr import jersey_crop, ocr_number
from src.pose_estimation import attach_keypoints_to_detections, get_pose_model, run_pose_on_frame
from src.post_process import build_field_mask, suppress_duplicate_tracks


def _box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter) / float(max(1, area_a + area_b - inter))


def run_pass1(
    source: Path,
    out_parquet: Path,
    *,
    clip_id: str | None = None,
    model_path: Path = SEG_MODEL_PATH,
    mask_store: Path = ROOT / "outputs" / "pass1_masks",
    pose_every: int = POSE_EVERY_DEFAULT,
    player_conf: float = PLAYER_PREDICT_CONF,
    player_iou: float = PLAYER_PREDICT_IOU,
    player_imgsz: int = PLAYER_IMGSZ,
    player_max_det: int = PLAYER_PREDICT_MAX_DET,
    helmet_model: Path | None = None,
    player_device: str | None = None,
) -> Path:
    if not source.exists():
        raise FileNotFoundError(source)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    mask_store.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    pose_model = get_pose_model() if pose_every > 0 else None
    helmet = YOLO(str(helmet_model)) if helmet_model else None
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    rows = []
    frame_idx = 0
    prev_frame = None
    cut_streak = 0
    stats = DetectionStats()
    cid = clip_id or source.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        is_cut = False
        if prev_frame is not None:
            mad = float(np.mean(np.abs(frame.astype(np.float32) - prev_frame.astype(np.float32))))
            if mad > CAMERA_CUT_MAD_THRESHOLD:
                cut_streak += 1
            else:
                cut_streak = 0
            if cut_streak >= CAMERA_CUT_CONSECUTIVE_FRAMES:
                is_cut = True
                cut_streak = 0

        field_mask = build_field_mask(frame)
        detections = extract_tracked_detections(
            model,
            frame,
            frame_idx=frame_idx,
            stats=stats,
            run_inference=True,
            apply_hud_filter=True,
            player_conf=player_conf,
            player_iou=player_iou,
            player_imgsz=player_imgsz,
            player_max_det=player_max_det,
            player_device=player_device,
            field_mask=field_mask,
        )
        detections = suppress_duplicate_tracks(detections)
        if pose_model is not None and frame_idx % max(1, pose_every) == 0 and detections:
            pose = run_pose_on_frame(pose_model, frame)
            attach_keypoints_to_detections(detections, pose)
        helmets = detect_helmets_roi(helmet, frame, detections) if helmet is not None else []

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            vec = build_raw_6d_vector(
                frame,
                det["mask"],
                (x1, y1, x2, y2),
                field_mask=field_mask,
                det=det,
                keypoints=det.get("keypoints"),
            )
            kpts = det.get("keypoints")
            kpts_list = kpts.tolist() if isinstance(kpts, np.ndarray) else None
            torso_px = int(np.count_nonzero(det["mask"]))
            mask_is_fallback = bool(det.get("mask_fallback", False))
            mask_ref = None
            if not mask_is_fallback:
                mask_name = f"{cid}_f{frame_idx:06d}_t{int(det['track_id']):04d}.npy"
                np.save(mask_store / mask_name, det["mask"])
                mask_ref = mask_name

            head_hits = 0
            for hb in helmets:
                if _box_iou((x1, y1, x2, y2), hb["bbox"]) >= 0.05:
                    head_hits += 1
            num, num_conf = ocr_number(jersey_crop(frame, (x1, y1, x2, y2)))

            rows.append(
                {
                    "clip_id": cid,
                    "frame": int(frame_idx),
                    "track_id": int(det["track_id"]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "conf": float(det["conf"]),
                    "mask_ref": mask_ref,
                    "mask_is_fallback": mask_is_fallback,
                    "feat_lab6d": vec.tolist() if vec is not None else None,
                    "torso_px": int(torso_px),
                    "kpts": kpts_list,
                    "helmet_hits": int(head_hits),
                    "ocr_num": int(num) if num is not None else None,
                    "ocr_conf": float(num_conf),
                    "is_cut_frame": bool(is_cut),
                    "team": None,
                }
            )

        prev_frame = frame
        frame_idx += 1

    cap.release()
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out_parquet)
    meta = {
        "source": str(source),
        "rows": len(rows),
        "frames": frame_idx,
        "clip_id": cid,
        "out_parquet": str(out_parquet),
        "mask_store": str(mask_store),
    }
    meta_path = out_parquet.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote IR rows={len(rows)} to {out_parquet}")
    return out_parquet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True, help="Parquet output path")
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--model", default=str(SEG_MODEL_PATH))
    parser.add_argument("--mask-store", default=str(ROOT / "outputs" / "pass1_masks"))
    parser.add_argument("--pose-every", type=int, default=POSE_EVERY_DEFAULT)
    parser.add_argument("--player-conf", type=float, default=PLAYER_PREDICT_CONF)
    parser.add_argument("--player-iou", type=float, default=PLAYER_PREDICT_IOU)
    parser.add_argument("--player-imgsz", type=int, default=PLAYER_IMGSZ)
    parser.add_argument("--player-max-det", type=int, default=PLAYER_PREDICT_MAX_DET)
    parser.add_argument("--helmet-model", default=None)
    args = parser.parse_args()
    run_pass1(
        Path(args.source),
        Path(args.out),
        clip_id=args.clip_id,
        model_path=Path(args.model),
        mask_store=Path(args.mask_store),
        pose_every=args.pose_every,
        player_conf=args.player_conf,
        player_iou=args.player_iou,
        player_imgsz=args.player_imgsz,
        player_max_det=args.player_max_det,
        helmet_model=Path(args.helmet_model) if args.helmet_model else None,
    )


if __name__ == "__main__":
    main()
