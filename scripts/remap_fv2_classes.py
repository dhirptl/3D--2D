"""Remap Football Videos-2 class IDs to the unified 3-class schema.

FV2 original (data.yaml order):
  0: ball      → 2 (ball)
  1: player    → 0 (player)
  2: players   → 0 (player)
  3: referee   → 1 (referee)
  4: team_A    → 0 (player)
  5: team_B    → 0 (player)

Target schema (combined_dataset/data.yaml):
  0: player
  1: referee
  2: ball

Edits labels in-place inside Football Videos-2/{train,valid,test}/labels/.
Run once before training.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FV2_ROOT = REPO_ROOT / "Football Videos-2"

# old_class_id → new_class_id  (None = drop this annotation)
_REMAP = {
    0: 2,    # ball → ball
    1: 0,    # player → player
    2: 0,    # players → player
    3: 1,    # referee → referee
    4: 0,    # team_A → player
    5: 0,    # team_B → player
}


def remap_label_file(path: Path) -> int:
    """Rewrite a single label file in-place. Returns number of lines written."""
    lines = path.read_text().strip().splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.split()
        old_cls = int(parts[0])
        new_cls = _REMAP.get(old_cls)
        if new_cls is None:
            continue
        out.append(f"{new_cls} " + " ".join(parts[1:]))
    path.write_text("\n".join(out))
    return len(out)


def main() -> None:
    splits = ["train", "valid", "test"]
    total_files = 0
    for split in splits:
        label_dir = FV2_ROOT / split / "labels"
        if not label_dir.exists():
            continue
        files = list(label_dir.glob("*.txt"))
        for f in files:
            remap_label_file(f)
        total_files += len(files)
        print(f"[remap_fv2] {split}: {len(files)} files remapped")

    print(f"[remap_fv2] Done. Total files: {total_files}")

    # Update data.yaml in-place
    data_yaml = FV2_ROOT / "data.yaml"
    if data_yaml.exists():
        import yaml
        with data_yaml.open() as f:
            cfg = yaml.safe_load(f)
        cfg["nc"] = 3
        cfg["names"] = ["player", "referee", "ball"]
        cfg.pop("roboflow", None)
        with data_yaml.open("w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"[remap_fv2] Updated {data_yaml}")


if __name__ == "__main__":
    main()
