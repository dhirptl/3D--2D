"""Fine-tune the segmentation model with new Football Videos-2 data.

Steps:
  1. Merge data/fv2_clean into merged_football_dataset_v2_seg (poly copy + bbox→rect)
  2. Fine-tune from run_v3-4_hardneg weights on the expanded dataset
  3. Save new run to football_tracker_seg/run_v3-5_fv2/

Usage:
  # Full pipeline (SAM generation + training):
  .venv/bin/python scripts/retrain.py

  # Skip SAM generation if already done:
  .venv/bin/python scripts/retrain.py --skip-sam

  # Dry run (print config, don't train):
  .venv/bin/python scripts/retrain.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import ROOT, SEG_MODEL_PATH

FV2_CLEAN = ROOT / "data" / "fv2_clean"
SEG_DATASET = ROOT / "merged_football_dataset_v2_seg"
BASE_WEIGHTS = SEG_MODEL_PATH  # run_v3-4_hardneg
RUN_NAME = "run_v3-5_fv2"
PROJECT = ROOT / "football_tracker_seg"


def run_fv2_merge() -> None:
    """Merge fv2_clean seg/bbox labels into the training dataset."""
    print("=" * 60)
    print("Step 1: Merge FV2 labels into seg dataset")
    print("=" * 60)
    if not FV2_CLEAN.exists():
        print(f"ERROR: {FV2_CLEAN} not found. Run scripts/dedup_fv2.py first.")
        sys.exit(1)

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "merge_fv2_seg.py")],
        cwd=str(ROOT),
        check=True,
    )
    print("FV2 merge complete.")


def run_training(dry_run: bool = False) -> None:
    """Fine-tune from hardneg weights on the expanded dataset."""
    print()
    print("=" * 60)
    print("Step 2: Fine-tune")
    print("=" * 60)

    n_train = len(list((SEG_DATASET / "train" / "images").glob("*.jpg")))
    n_val   = len(list((SEG_DATASET / "val"   / "images").glob("*.jpg")))

    config = dict(
        task="segment",
        mode="train",
        model=str(BASE_WEIGHTS),        # fine-tune from hardneg
        data=str(SEG_DATASET / "data.yaml"),
        epochs=15,                       # first-pass fine-tune (508 new images)
        patience=15,
        batch=8,
        imgsz=1024,
        device="mps",
        workers=0,                       # required for MPS stability
        project=str(PROJECT),
        name=RUN_NAME,
        exist_ok=False,
        amp=True,
        close_mosaic=5,
        seed=0,
    )

    print(f"  Base weights : {BASE_WEIGHTS}")
    print(f"  Dataset      : train={n_train} images, val={n_val} images")
    print(f"  Output       : {PROJECT}/{RUN_NAME}/")
    print(f"  Epochs       : {config['epochs']} (fine-tune, not full retrain)")
    print(f"  Device       : {config['device']}")
    print(f"  Est. time    : ~1-2 hours on M4 Pro")
    print()

    if dry_run:
        print("Dry run — config printed above. Pass no --dry-run flag to train.")
        return

    from ultralytics import YOLO
    model = YOLO(str(BASE_WEIGHTS))
    model.train(**config)

    best = PROJECT / RUN_NAME / "weights" / "best.pt"
    print(f"\nTraining complete. Best weights: {best}")
    print()
    print("Evaluate with:")
    print(f"  .venv/bin/python scripts/eval_detection.py \\")
    print(f"      --weights {best} \\")
    print(f"      --data data/football_eval/data.yaml")
    print()
    print("Update config.py to use new model:")
    print(f"  SEG_MODEL_PATH = ROOT / 'football_tracker_seg' / '{RUN_NAME}' / 'weights' / 'best.pt'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sam", action="store_true",
                        help="Skip FV2 merge (if already done)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config only, don't train")
    args = parser.parse_args()

    if not args.skip_sam and not args.dry_run:
        run_fv2_merge()

    run_training(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
