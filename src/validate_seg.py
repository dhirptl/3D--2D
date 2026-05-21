"""Gate 2: validate seg model on football_dataset_seg."""

import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import (
    GATE2_MASK_MAP50,
    GATE2_MASK_RECALL,
    SEG_DATASET_YAML,
    SEG_MODEL_PATH,
)


def validate(weights: Path | None = None) -> bool:
    w = weights or SEG_MODEL_PATH
    if not w.exists():
        raise FileNotFoundError(f"Weights not found: {w}")
    if not SEG_DATASET_YAML.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {SEG_DATASET_YAML}")

    model = YOLO(str(w))
    metrics = model.val(data=str(SEG_DATASET_YAML), verbose=False)

    def scalar(val) -> float:
        arr = np.asarray(val)
        return float(arr.mean() if arr.size > 1 else arr.item())

    box_map50 = scalar(metrics.box.map50)
    box_recall = scalar(metrics.box.r)
    mask_map50 = scalar(metrics.seg.map50)
    mask_recall = scalar(metrics.seg.r)

    print("--- Gate 2 Validation ---")
    box_mp = scalar(metrics.box.mp)
    mask_mp = scalar(metrics.seg.mp)
    print(f"  Box  P/R/mAP50: {box_mp:.3f} / {box_recall:.3f} / {box_map50:.3f}")
    print(f"  Mask P/R/mAP50: {mask_mp:.3f} / {mask_recall:.3f} / {mask_map50:.3f}")

    passed = mask_map50 >= GATE2_MASK_MAP50 and mask_recall >= GATE2_MASK_RECALL
    if passed:
        print(f"  PASSED (mask mAP50>={GATE2_MASK_MAP50}, recall>={GATE2_MASK_RECALL})")
    else:
        print(f"  FAILED (need mask mAP50>={GATE2_MASK_MAP50}, recall>={GATE2_MASK_RECALL})")
    return passed


if __name__ == "__main__":
    ok = validate()
    sys.exit(0 if ok else 1)
