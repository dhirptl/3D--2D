"""End-to-end seg detect + track + team classification."""

import argparse
import csv
import pickle
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import (
    BYTETRACK_CFG,
    CALIBRATION_VERSION,
    DETECT_EVERY_DEFAULT,
    HELMET_CONF,
    HELMET_GATE_GRACE,
    HELMET_MODEL_PATH,
    PIPELINE_THREADS_DEFAULT,
    PLAYER_IMGSZ,
    PLAYER_PREDICT_CONF,
    PLAYER_PREDICT_IOU,
    PLAYER_PREDICT_MAX_DET,
    POSE_EVERY_DEFAULT,
    RETINA_BENCHMARK_FRAMES,
    RETINA_MAX_MS_PER_FRAME,
    ROOT as CONFIG_ROOT,
    SEG_MODEL_PATH,
    TRACKER_CFG,
)
from src.detection import DetectionStats, extract_tracked_detections
from src.draw import draw_teams
from src.mask_utils import mask_area
from src.pipeline import VideoPipelineContext, process_video_frame
from src.team_classifier import FootballTeamClassifier
from src.video_pipeline import run_with_read_ahead


def load_calibration(classifier: FootballTeamClassifier, path: Path) -> None:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load calibration: {e}") from e
    classifier.load_calibration(data)
    print(f"Loaded calibration from {path} (v{data.get('version', 0)})")


def _open_video_writer(out_path: Path, fps: float, size: tuple[int, int], codec: str):
    fourcc = cv2.VideoWriter_fourcc(*codec[:4].ljust(4))
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    if writer.isOpened():
        return writer
    if codec != "mp4v":
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, size)
    return writer


def run(
    source: str,
    out: str,
    model_path: Path | None = None,
    *,
    use_team: bool = True,
    filter_field: bool = False,
    apply_hud_filter: bool = True,
    show_masks: bool = False,
    debug_teams: bool = False,
    save_calibration: str | None = None,
    load_calibration_path: str | None = None,
    dump_detections: str | None = None,
    detect_every: int = DETECT_EVERY_DEFAULT,
    codec: str = "avc1",
    pipeline_threads: int = PIPELINE_THREADS_DEFAULT,
    retina_masks: bool = False,
    hue_weight: float | None = None,
    force_retina: bool = False,
    no_prefetch: bool = False,
    tracker: str = "botsort",
    pose_every: int = POSE_EVERY_DEFAULT,
    no_pose: bool = False,
    require_helmet: bool = False,
    helmet_model_path: Path | None = None,
    helmet_conf: float = HELMET_CONF,
    use_helmet_gate: bool = True,
    helmet_gate_grace: int = HELMET_GATE_GRACE,
    player_conf: float = PLAYER_PREDICT_CONF,
    player_iou: float = PLAYER_PREDICT_IOU,
    player_imgsz: int = PLAYER_IMGSZ,
    player_max_det: int = PLAYER_PREDICT_MAX_DET,
) -> None:
    weights = model_path or SEG_MODEL_PATH
    if not weights.exists():
        raise FileNotFoundError(
            f"Seg model not found: {weights}. Train: python -m src.train_football_seg"
        )
    helmet_weights = helmet_model_path or HELMET_MODEL_PATH
    if require_helmet and not helmet_weights.exists():
        raise FileNotFoundError(
            "Helmet model not found: "
            f"{helmet_weights}. Run: python -m src.prepare_helmet_dataset "
            "&& python -m src.train_helmet_detector"
        )

    model = YOLO(str(weights))
    classifier = (
        FootballTeamClassifier(hue_weight=hue_weight) if use_team else None
    )
    tracker_path = TRACKER_CFG if tracker == "botsort" else BYTETRACK_CFG
    ctx = VideoPipelineContext(
        model=model,
        classifier=classifier,
        apply_hud_filter=apply_hud_filter,
        filter_field=filter_field,
        detect_every=max(1, detect_every),
        retina_masks=retina_masks,
        tracker_cfg=tracker_path,
        player_conf=player_conf,
        player_iou=player_iou,
        player_imgsz=max(64, int(player_imgsz)),
        player_max_det=max(1, int(player_max_det)),
        pose_every=max(1, pose_every),
        use_pose=not no_pose,
        require_helmet=require_helmet,
        helmet_model_path=helmet_weights,
        helmet_conf=helmet_conf,
        use_helmet_gate=use_helmet_gate,
        helmet_gate_grace=max(0, helmet_gate_grace),
    )
    team_state_counts: Counter = Counter()
    total_frames = 0

    if load_calibration_path and classifier is not None:
        load_calibration(classifier, Path(load_calibration_path))

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    use_retina = retina_masks
    if use_retina and not force_retina:
        warmup_times: list[float] = []
        for _ in range(RETINA_BENCHMARK_FRAMES):
            ret, bench_frame = cap.read()
            if not ret:
                break
            t0 = time.perf_counter()
            extract_tracked_detections(
                model,
                bench_frame,
                frame_idx=0,
                run_inference=True,
                retina_masks=True,
                player_conf=ctx.player_conf,
                player_iou=ctx.player_iou,
                player_imgsz=ctx.player_imgsz,
                player_max_det=ctx.player_max_det,
            )
            warmup_times.append(time.perf_counter() - t0)
        if warmup_times:
            avg_ms = 1000 * sum(warmup_times) / len(warmup_times)
            if avg_ms > RETINA_MAX_MS_PER_FRAME:
                print(
                    f"[perf] retina_masks disabled ({avg_ms:.0f}ms/frame > "
                    f"{RETINA_MAX_MS_PER_FRAME:.0f}ms limit)"
                )
                use_retina = False
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ctx.retina_masks = use_retina

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = _open_video_writer(out_path, fps, (w, h), codec)
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open video writer for {out_path}")

    dump_file = None
    dump_writer = None
    if dump_detections:
        dp = Path(dump_detections)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dump_file = dp.open("w", newline="")
        dump_writer = csv.writer(dump_file)
        dump_writer.writerow([
            "frame",
            "track_id",
            "conf",
            "box_w",
            "box_h",
            "mask_area",
            "classifier_state",
            "team_label",
            "dist0",
            "dist1",
            "margin",
            "warmup_rejected",
        ])

    def handle_frame(frame: np.ndarray, frame_idx: int) -> np.ndarray:
        nonlocal total_frames
        total_frames += 1
        detections, team_labels = process_video_frame(frame, ctx)

        if classifier is not None:
            team_state_counts[classifier.state] += 1
            annotated = draw_teams(
                frame,
                detections,
                team_labels or {},
                classifier,
                show_masks=show_masks,
                debug_teams=debug_teams,
            )
        else:
            annotated = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if dump_writer is not None:
            state = classifier.state if classifier else "none"
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                tid = det["track_id"]
                lab = (team_labels or {}).get(tid, -1)
                dbg = classifier.last_frame_debug.get(tid, {}) if classifier else {}
                dump_writer.writerow([
                    frame_idx,
                    tid,
                    det["conf"],
                    x2 - x1,
                    y2 - y1,
                    mask_area(det["mask"]),
                    state,
                    lab,
                    dbg.get("dist0", ""),
                    dbg.get("dist1", ""),
                    dbg.get("margin", ""),
                    dbg.get("reject", ""),
                ])

        return annotated

    if pipeline_threads > 0 and not no_prefetch:
        frame_idx = run_with_read_ahead(
            cap,
            lambda frame, fidx: handle_frame(frame, fidx),
            writer.write,
            queue_size=max(2, pipeline_threads),
        )
    else:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = handle_frame(frame, frame_idx)
            writer.write(annotated)
            frame_idx += 1

    cap.release()
    writer.release()
    if dump_file is not None:
        dump_file.close()

    if save_calibration and classifier is not None and classifier.state == FootballTeamClassifier.STATE_LOCKED:
        cal_path = Path(save_calibration)
        cal_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cal_path, "wb") as f:
            pickle.dump(classifier.calibration_dict(), f)
        print(f"Saved calibration to {cal_path} (v{CALIBRATION_VERSION})")

    print(ctx.det_stats.summary())
    if ctx.det_stats.frames:
        zero_det_rate = ctx.det_stats.zero_det_frames / ctx.det_stats.frames
        print(f"Zero-det rate: {zero_det_rate:.1%}")
    print(f"Wrote {frame_idx} frames to {out_path}")
    if total_frames:
        print(f"Track ID change rate: {ctx.track_id_changes / total_frames:.4f}")
        print(f"Registration valid frames: {100 * ctx.registration_valid_frames / total_frames:.1f}%")
    if classifier is not None:
        print(f"Classifier states: {dict(team_state_counts)}")
        print(f"Final state: {classifier.state}  warmup_samples: {classifier.warmup_count}")
        if classifier.locked_frame_index is not None and classifier.locked_frame_index >= 0:
            print(f"First LOCKED frame: {classifier.locked_frame_index}")
        if total_frames:
            locked_pct = 100 * team_state_counts.get(FootballTeamClassifier.STATE_LOCKED, 0) / total_frames
            print(f"LOCKED frames: {locked_pct:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Football seg + track + teams")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=str(CONFIG_ROOT / "outputs" / "tracked.mp4"))
    parser.add_argument("--model", default=None)
    parser.add_argument("--no-team", action="store_true")
    parser.add_argument(
        "--filter-field",
        action="store_true",
        help="Require playable-area overlap near the feet (homography-backed with HSV fallback)",
    )
    parser.add_argument("--no-hud-filter", action="store_true")
    parser.add_argument("--show-masks", action="store_true")
    parser.add_argument("--debug-teams", action="store_true", help="Show dist0/dist1 and reject reasons")
    parser.add_argument("--retina-masks", action="store_true", help="Higher-res seg masks (slower)")
    parser.add_argument("--save-calibration", default=None)
    parser.add_argument("--load-calibration", default=None)
    parser.add_argument("--dump-detections", default=None)
    parser.add_argument("--detect-every", type=int, default=DETECT_EVERY_DEFAULT)
    parser.add_argument("--codec", default="avc1", help="Video fourcc (avc1 or mp4v)")
    parser.add_argument(
        "--pipeline-threads",
        type=int,
        default=PIPELINE_THREADS_DEFAULT,
        help="Read-ahead decode threads (0 disables prefetch)",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable async frame prefetch",
    )
    parser.add_argument(
        "--hue-weight",
        type=float,
        default=None,
        help="Deprecated (LAB features); ignored",
    )
    parser.add_argument(
        "--tracker",
        choices=("botsort", "bytetrack"),
        default="botsort",
        help="Tracker config (default: botsort with Re-ID)",
    )
    parser.add_argument(
        "--player-conf",
        type=float,
        default=PLAYER_PREDICT_CONF,
        help="Player detector confidence threshold",
    )
    parser.add_argument(
        "--player-iou",
        type=float,
        default=PLAYER_PREDICT_IOU,
        help="Player detector NMS IoU threshold",
    )
    parser.add_argument(
        "--player-imgsz",
        type=int,
        default=PLAYER_IMGSZ,
        help="Player detector inference image size",
    )
    parser.add_argument(
        "--player-max-det",
        type=int,
        default=PLAYER_PREDICT_MAX_DET,
        help="Player detector max detections per frame",
    )
    parser.add_argument(
        "--pose-every",
        type=int,
        default=POSE_EVERY_DEFAULT,
        help="Run pose model every N frames (1=every frame)",
    )
    parser.add_argument(
        "--no-pose",
        action="store_true",
        help="Disable YOLO-Pose torso sampling",
    )
    parser.add_argument(
        "--force-retina",
        action="store_true",
        help="Keep retina_masks even if benchmark is slow",
    )
    parser.add_argument(
        "--require-helmet",
        action="store_true",
        help="Keep only tracked detections with a matching helmet near the head region",
    )
    parser.add_argument(
        "--helmet-model",
        default=None,
        help="Override helmet detector weights path",
    )
    parser.add_argument(
        "--helmet-conf",
        type=float,
        default=HELMET_CONF,
        help="Helmet detector confidence threshold",
    )
    parser.add_argument(
        "--no-helmet-gate",
        action="store_true",
        help="Disable temporal smoothing for helmet verification",
    )
    parser.add_argument(
        "--helmet-gate-grace",
        type=int,
        default=HELMET_GATE_GRACE,
        help="Allow this many initial frames before helmet confirmation is required",
    )
    args = parser.parse_args()

    run(
        args.source,
        args.out,
        Path(args.model) if args.model else None,
        use_team=not args.no_team,
        filter_field=args.filter_field,
        apply_hud_filter=not args.no_hud_filter,
        show_masks=args.show_masks,
        debug_teams=args.debug_teams,
        save_calibration=args.save_calibration,
        load_calibration_path=args.load_calibration,
        dump_detections=args.dump_detections,
        detect_every=args.detect_every,
        codec=args.codec,
        pipeline_threads=args.pipeline_threads,
        retina_masks=args.retina_masks,
        hue_weight=args.hue_weight,
        force_retina=args.force_retina,
        no_prefetch=args.no_prefetch,
        tracker=args.tracker,
        pose_every=args.pose_every,
        no_pose=args.no_pose,
        require_helmet=args.require_helmet,
        helmet_model_path=Path(args.helmet_model) if args.helmet_model else None,
        helmet_conf=args.helmet_conf,
        use_helmet_gate=not args.no_helmet_gate,
        helmet_gate_grace=args.helmet_gate_grace,
        player_conf=args.player_conf,
        player_iou=args.player_iou,
        player_imgsz=args.player_imgsz,
        player_max_det=args.player_max_det,
    )


if __name__ == "__main__":
    main()
