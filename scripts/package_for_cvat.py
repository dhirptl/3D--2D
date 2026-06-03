"""Package data/broadcast_eval into a zip CVAT can import directly.

CVAT YOLO 1.1 import format:
  upload_for_cvat.zip
  ├── obj.data          (metadata)
  ├── obj.names         (class names)
  ├── train.txt         (image paths)
  └── obj_train_data/
      ├── frame_000.jpg
      ├── frame_000.txt  (YOLO boxes: class cx cy w h, normalised)
      └── ...

Usage:
  python scripts/package_for_cvat.py
  -> writes data/broadcast_eval_cvat.zip

Then on app.cvat.ai:
  1. Create project "football-broadcast-eval", label "Player" (type: rectangle)
  2. Create task -> upload images from obj_train_data/ only
     (or upload the whole zip via "YOLO 1.1" import which pre-loads boxes)
  3. Correct the pre-labels (add missed players, remove FPs, fix bad boxes)
  4. Export task -> "YOLO 1.1" format -> download zip
  5. Run: python scripts/import_cvat_labels.py <downloaded.zip>
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

from src.config import ROOT

EVAL_ROOT = ROOT / "data" / "broadcast_eval"
OUT_ZIP = ROOT / "data" / "broadcast_eval_cvat.zip"
CLASS_NAMES = ["Player"]


def package(eval_root: Path = EVAL_ROOT, out_zip: Path = OUT_ZIP) -> None:
    img_dir = eval_root / "images"
    lbl_dir = eval_root / "labels"
    if not img_dir.exists():
        raise FileNotFoundError(f"Run build_eval_set.py first: {img_dir} not found")

    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        raise RuntimeError(f"No images found in {img_dir}")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # obj.names
        zf.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")

        # obj.data
        zf.writestr(
            "obj.data",
            f"classes = {len(CLASS_NAMES)}\n"
            f"train = data/train.txt\n"
            f"names = data/obj.names\n"
            f"backup = backup/\n",
        )

        # train.txt and images+labels in obj_train_data/
        train_lines = []
        for img_path in images:
            name = img_path.name
            stem = img_path.stem
            arc_img = f"obj_train_data/{name}"
            arc_lbl = f"obj_train_data/{stem}.txt"

            zf.write(img_path, arc_img)
            train_lines.append(f"data/{arc_img}")

            lbl_path = lbl_dir / f"{stem}.txt"
            if lbl_path.exists():
                zf.write(lbl_path, arc_lbl)
            else:
                zf.writestr(arc_lbl, "")  # empty = no detections on this frame

        zf.writestr("train.txt", "\n".join(train_lines) + "\n")

    print(f"Wrote {len(images)} frames to {out_zip} ({out_zip.stat().st_size // 1024} KB)")
    print()
    print("Next steps:")
    print("  1. Go to https://app.cvat.ai")
    print("  2. Create project: 'football-broadcast-eval'")
    print("     Label: 'Player' (rectangle)")
    print(f"  3. Create task -> 'Import data' -> upload {out_zip.name}")
    print("     Format: YOLO 1.1 (pre-loads the model's box predictions as annotations)")
    print("  4. Open task, correct boxes: add missed players, delete FPs, tighten loose boxes")
    print("  5. Export annotations -> Format: YOLO 1.1 -> download zip")
    print("  6. Run: python scripts/import_cvat_labels.py <downloaded_zip>")


if __name__ == "__main__":
    package()
