"""Package data/broadcast_eval into a zip Roboflow can import directly.

Roboflow YOLO upload format (simplest path):
  broadcast_eval_roboflow.zip
  ├── images/
  │   ├── frame_000.jpg
  │   └── ...
  └── labels/
      ├── frame_000.txt   (YOLO: class cx cy w h, normalised)
      └── ...

Usage:
  python scripts/package_for_cvat.py
  -> writes data/broadcast_eval_roboflow.zip

Then on app.roboflow.com:
  1. Create project: Object Detection, "football-broadcast-eval"
  2. Upload -> drag-drop broadcast_eval_roboflow.zip
     Roboflow auto-detects YOLO format and pre-loads box annotations.
  3. Correct pre-labels: add missed players, delete FPs, tighten loose boxes
  4. Generate dataset version (no augmentation needed for eval set)
  5. Export -> Format: YOLOv8 -> download zip
  6. Run: python scripts/import_cvat_labels.py <downloaded_zip>
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.config import ROOT

EVAL_ROOT = ROOT / "data" / "broadcast_eval"
OUT_ZIP = ROOT / "data" / "broadcast_eval_roboflow.zip"


def package(eval_root: Path = EVAL_ROOT, out_zip: Path = OUT_ZIP) -> None:
    img_dir = eval_root / "images"
    lbl_dir = eval_root / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"Run build_eval_set.py first: {img_dir} not found")

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for img_path in images:
            zf.write(img_path, f"images/{img_path.name}")
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                zf.write(lbl_path, f"labels/{lbl_path.name}")
            else:
                zf.writestr(f"labels/{img_path.stem}.txt", "")

    print(f"Wrote {len(images)} frames to {out_zip} ({out_zip.stat().st_size // 1024} KB)")
    print()
    print("Next steps:")
    print("  1. Go to https://app.roboflow.com")
    print("  2. New project -> Object Detection -> 'football-broadcast-eval'")
    print(f"  3. Upload -> drag-drop {out_zip.name}")
    print("     Roboflow detects YOLO format and pre-loads box annotations.")
    print("  4. Correct boxes: add missed players, delete FPs, tighten loose boxes")
    print("  5. Versions -> Generate -> Export -> YOLOv8 -> download zip")
    print("  6. Run: python scripts/import_cvat_labels.py <downloaded_zip>")


if __name__ == "__main__":
    package()
