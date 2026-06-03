"""Import corrected CVAT YOLO 1.1 export back into data/broadcast_eval/labels/.

Usage:
  python scripts/import_cvat_labels.py <cvat_export.zip>

The CVAT YOLO 1.1 export zip contains obj_train_data/*.txt (one per image).
This script extracts those label files and writes them to
data/broadcast_eval/labels/, replacing the pre-labels from build_eval_set.py.

After running this, score with:
  .venv/bin/python -m src.validate_seg \\
      --weights football_tracker_seg/run_v3-4_hardneg/weights/best.pt \\
      --data data/broadcast_eval/data.yaml
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

from src.config import ROOT

EVAL_ROOT = ROOT / "data" / "broadcast_eval"


def import_labels(zip_path: Path, eval_root: Path = EVAL_ROOT) -> None:
    lbl_dir = eval_root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [m for m in zf.namelist() if m.endswith(".txt") and "obj_train_data" in m]
        if not txt_members:
            # Try flat structure (some CVAT versions)
            txt_members = [m for m in zf.namelist() if m.endswith(".txt") and "obj." not in m and "train" not in m]
        if not txt_members:
            raise RuntimeError(f"No label .txt files found in {zip_path}. Members: {zf.namelist()[:10]}")

        for member in txt_members:
            stem = Path(member).stem
            dest = lbl_dir / f"{stem}.txt"
            with zf.open(member) as src, dest.open("wb") as dst:
                dst.write(src.read())
            imported += 1

    print(f"Imported {imported} label files to {lbl_dir}")
    print()
    print("Score with:")
    print("  .venv/bin/python -m src.validate_seg \\")
    print(f"      --weights football_tracker_seg/run_v3-4_hardneg/weights/best.pt \\")
    print(f"      --data {eval_root}/data.yaml")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/import_cvat_labels.py <cvat_export.zip>")
        sys.exit(1)
    import_labels(Path(sys.argv[1]))
