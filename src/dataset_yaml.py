"""Validate YOLO data.yaml files before training."""

from pathlib import Path

import yaml

TASK_EXPECTATIONS = {
    "seg": {"nc": 1, "names": {"player", "Player"}},
    "helmet": {"nc": 1, "names": {"helmet"}},
}


def validate_dataset_yaml(path: Path, task: str) -> Path:
    """Raise ValueError if yaml does not match the expected single-class task schema."""
    if task not in TASK_EXPECTATIONS:
        raise ValueError(f"Unknown task {task!r}; expected one of {list(TASK_EXPECTATIONS)}")

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {path}")

    cfg = yaml.safe_load(path.read_text()) or {}
    exp = TASK_EXPECTATIONS[task]
    nc = cfg.get("nc")
    if nc != exp["nc"]:
        raise ValueError(
            f"{path.name}: expected nc={exp['nc']} for task={task}, got nc={nc}. "
            "Do not use the legacy root data.yaml (nc=2 ball/player) for seg/helmet training."
        )

    names = cfg.get("names") or []
    if isinstance(names, dict):
        name_set = {str(v).lower() for v in names.values()}
    else:
        name_set = {str(n).lower() for n in names}
    if not name_set & exp["names"]:
        raise ValueError(
            f"{path.name}: class names {names!r} do not match task={task} "
            f"(expected one of {sorted(exp['names'])})"
        )

    root = Path(cfg.get("path", ".")).resolve()
    train_rel = cfg.get("train")
    if train_rel:
        train_dir = root / train_rel
        if not train_dir.exists():
            raise FileNotFoundError(f"Train images missing: {train_dir}")

    val_rel = cfg.get("val")
    if val_rel:
        val_dir = root / val_rel
        if not val_dir.exists():
            raise FileNotFoundError(f"Val images missing: {val_dir}")

    return path
