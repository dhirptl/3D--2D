"""Prepare Roboflow export: single-class Player, stratified 80/20 split, Gate 1."""

import shutil
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "train" / "images"
RAW_LABELS = ROOT / "train" / "labels"
OUT_ROOT = ROOT / "football_dataset"
PLAYER_CLASS_RAW = 1
TEST_SIZE = 0.2
RANDOM_STATE = 42


def player_bin(count: int) -> str:
    if count <= 6:
        return "low"
    if count <= 14:
        return "mid"
    return "high"


def convert_label(src: Path) -> list[str]:
    lines = []
    for line in src.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if int(parts[0]) != PLAYER_CLASS_RAW:
            continue
        parts[0] = "0"
        lines.append(" ".join(parts))
    return lines


def gate1(train_labels: Path, val_labels: Path, data_yaml: Path) -> None:
    cfg = yaml.safe_load(data_yaml.read_text())
    assert cfg["nc"] == 1, f"Gate 1 fail: nc={cfg['nc']}, expected 1"
    assert cfg["names"] == ["Player"], f"Gate 1 fail: names={cfg['names']}"

    train_stems = {p.stem for p in train_labels.glob("*.txt")}
    val_stems = {p.stem for p in val_labels.glob("*.txt")}
    assert not train_stems & val_stems, "Gate 1 fail: train/val overlap"

    for split_name, labels_dir in [("train", train_labels), ("valid", val_labels)]:
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().strip().splitlines():
                if line.strip():
                    cls = int(line.split()[0])
                    assert cls == 0, f"Gate 1 fail: class {cls} in {label_file}"

    n_train, n_val = len(train_stems), len(val_stems)
    print(f"Gate 1 PASSED: nc=1, train={n_train}, valid={n_val}, disjoint splits")


def main() -> None:
    if not RAW_IMAGES.exists():
        raise FileNotFoundError(f"Raw images not found: {RAW_IMAGES}")

    stems = []
    bins = []
    converted: dict[str, list[str]] = {}

    for label_path in sorted(RAW_LABELS.glob("*.txt")):
        stem = label_path.stem
        img_path = RAW_IMAGES / f"{stem}.jpg"
        if not img_path.exists():
            for ext in (".png", ".jpeg"):
                alt = RAW_IMAGES / f"{stem}{ext}"
                if alt.exists():
                    img_path = alt
                    break
            else:
                print(f"Skip {stem}: no matching image")
                continue

        lines = convert_label(label_path)
        if not lines:
            print(f"Skip {stem}: no player annotations")
            continue

        converted[stem] = lines
        stems.append(stem)
        bins.append(player_bin(len(lines)))

    train_stems, val_stems = train_test_split(
        stems, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=bins
    )

    for split, split_stems in [("train", train_stems), ("valid", val_stems)]:
        img_dir = OUT_ROOT / split / "images"
        lbl_dir = OUT_ROOT / split / "labels"
        if img_dir.exists():
            shutil.rmtree(img_dir.parent)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for stem in split_stems:
            src_img = RAW_IMAGES / f"{stem}.jpg"
            if not src_img.exists():
                for ext in (".png", ".jpeg"):
                    alt = RAW_IMAGES / f"{stem}{ext}"
                    if alt.exists():
                        src_img = alt
                        break
            shutil.copy2(src_img, img_dir / src_img.name)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(converted[stem]) + "\n")

    data = {
        "path": str(OUT_ROOT.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["Player"],
    }
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))

    gate1(OUT_ROOT / "train" / "labels", OUT_ROOT / "valid" / "labels", yaml_path)
    print(f"Dataset written to {OUT_ROOT}")
    print(f"  train: {len(train_stems)}, valid: {len(val_stems)}")


if __name__ == "__main__":
    main()
