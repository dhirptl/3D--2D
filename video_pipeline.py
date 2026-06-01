"""Root-level wrapper: run the detect+track+team pipeline on an eval clip.

Usage:
    python video_pipeline.py --clip test_clips/eval_clip.mp4

Produces:
    outputs/eval_latest.mp4         annotated video
    outputs/eval_detections.csv     per-frame detection dump consumed by evaluate.py

All actual work is done by src.run_tracker.run().
"""

import argparse

from src.run_tracker import run


def main() -> None:
    ap = argparse.ArgumentParser(description="Run football tracker pipeline on an eval clip")
    ap.add_argument("--clip", required=True, help="Input video path")
    ap.add_argument("--out", default="outputs/eval_latest.mp4", help="Annotated output video")
    ap.add_argument(
        "--dump-detections",
        default="outputs/eval_detections.csv",
        help="Per-frame detection CSV for evaluate.py route metrics",
    )
    args = ap.parse_args()

    run(
        args.clip,
        args.out,
        dump_detections=args.dump_detections,
    )


if __name__ == "__main__":
    main()
