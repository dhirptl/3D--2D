"""Import corrected Roboflow YOLOv8 export back into data/broadcast_eval/labels/.

Usage:
  python scripts/import_cvat_labels.py <roboflow_export.zip>

Roboflow's YOLOv8 export zip structure:
  ├── data.yaml
  ├── train/images/*.jpg  and  train/labels/*.txt
  └── valid/images/*.jpg  and  valid/labels/*.txt

This script collects all .txt label files from every split and writes them
to data/broadcast_eval/labels/, replacing the pre-labels from build_eval_set.py.

After running this, score with:
  .venv/bin/python -m src.validate_seg \\
      --weights football_tracker_seg/run_v3-4_hardneg/weights/best.pt \\
      --data data/broadcast_eval/data.yaml
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from src.config import ROOT

EVAL_ROOT = ROOT / "data" / "broadcast_eval"

# Roboflow YOLOv8 export puts labels under <split>/labels/*.txt
# Collect from all splits so train/valid/test are all captured.
_LABEL_MARKERS = ("labels/",)
_SKIP = ("data.yaml", "README")


def import_labels(zip_path: Path, eval_root: Path = EVAL_ROOT) -> None:
    lbl_dir = eval_root / "labels"
    lbl_dir.mkdir(parents=True, exist_ok=True)

    imported = 0
    with zipfile.ZipFile(zip_path) as zf:
        txt_members = [
            m for m in zf.namelist()
            if m.endswith(".txt")
            and any(marker in m for marker in _LABEL_MARKERS)
            and not any(s in m for s in _SKIP)
        ]
        if not txt_members:
            # Fallback: any .txt that isn't a manifest
            txt_members = [
                m for m in zf.namelist()
                if m.endswith(".txt") and not any(s in m for s in _SKIP)
            ]
        if not txt_members:
            raise RuntimeError(
                f"No label .txt files found in {zip_path}.\n"
                f"Members: {zf.namelist()[:15]}"
            )

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
