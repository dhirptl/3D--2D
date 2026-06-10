#!/usr/bin/env python3
"""Phase 0: frame-level flow for collapsed-valid cohort C vs post-guard pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIELD_REG_UPDATE_STRIDE_STABLE, FIELD_REG_STABLE_STREAK
from src.field_registration import FieldRegistration, _projection_area_frac
from zerodet_decompose import bucket_raw, load_jsonl, merge_pipeline, raw_counts

MONTAGE = ROOT / "1minclip.mov"
CONTINUOUS = ROOT / "outputs" / "arz_nyj_1min.mov"
OUT_MD = ROOT / "outputs" / "phase0_reconciliation.md"
OUT_JSON = ROOT / "outputs" / "phase0_reconciliation.json"
CONF = 0.18


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _replay_cohort(source: Path, *, disable_guard: bool) -> tuple[list[int], dict[int, float], int, int]:
    """Frames where registration_valid and area_frac < 0.08."""
    import src.field_registration as fr

    orig = fr._projection_is_plausible
    if disable_guard:
        fr._projection_is_plausible = lambda H, h, w: True  # noqa: ARG005

    reg = FieldRegistration()
    cap = cv2.VideoCapture(str(source))
    cohort: list[int] = []
    area_by_frame: dict[int, float] = {}
    reg_valid_n = 0
    total = 0
    fi = 0
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            fh, fw = frame.shape[:2]
            streak = reg.valid_streak
            stride = (
                FIELD_REG_UPDATE_STRIDE_STABLE
                if streak >= FIELD_REG_STABLE_STREAK
                else 1
            )
            if fi % stride == 0:
                reg.update(frame)
            if reg.registration_valid:
                reg_valid_n += 1
                if reg.homography is not None:
                    frac = _projection_area_frac(reg.homography, fh, fw)
                    if frac is not None and frac < 0.08:
                        cohort.append(fi)
                        area_by_frame[fi] = frac
            total += 1
            fi += 1
    finally:
        fr._projection_is_plausible = orig
        cap.release()
    return cohort, area_by_frame, reg_valid_n, total


def _classify_outcome(rec: dict | None, raw_n: int) -> str:
    if rec is None:
        return "tracked_ok"
    path = rec.get("path", "OK")
    if path == "HUD_ATE_ALL":
        return "HUD_ATE_ALL"
    if path != "ID_NONE":
        return "other_zero"
    rb = rec.get("raw_boxes", raw_n)
    if rb == 0:
        return "ID_NONE_raw0"
    if rb <= 2:
        return "ID_NONE_sparse"
    return "ID_NONE_boundary"


def _flow_for_cohort(
    cohort: list[int],
    jsonl_by_frame: dict,
    per_frame_raw: list[tuple[int, int, float]],
) -> Counter:
    raw_map = {i: n for i, n, _ in per_frame_raw}
    counts: Counter = Counter()
    for fi in cohort:
        counts[_classify_outcome(jsonl_by_frame.get(fi), raw_map.get(fi, 0))] += 1
    return counts


def _hud_fixed_band(jsonl_by_frame: dict, frames: list[int]) -> int:
    n = 0
    for fi in frames:
        rec = jsonl_by_frame.get(fi)
        if rec and rec.get("path") == "HUD_ATE_ALL":
            fm = rec.get("field_mask", "")
            if fm == "none":
                n += 1
    return n


def _run_tracker_jsonl(source: Path, out_jsonl: Path, *, disable_guard: bool) -> None:
    env = {**dict(subprocess.os.environ), "ZERODET_DEBUG": str(out_jsonl)}
    if disable_guard:
        env["FIELD_REG_DISABLE_PROJ_GUARD"] = "1"
    else:
        env.pop("FIELD_REG_DISABLE_PROJ_GUARD", None)
    out_mp4 = out_jsonl.with_suffix(".mp4")
    cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        "-m",
        "src.run_tracker",
        "--source",
        str(source),
        "--out",
        str(out_mp4),
        "--tracker",
        "bytetrack",
        "--no-pass1-gate",
    ]
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    head = _git_head()
    try:
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", "fc4a5eb", "HEAD"], cwd=ROOT
        )
        guard_ancestor = True
    except subprocess.CalledProcessError:
        guard_ancestor = False

    post_jsonl = ROOT / "outputs" / "zerodet_post.jsonl"
    pre_jsonl = ROOT / "outputs" / "zerodet_pre_noguard.jsonl"

    if not post_jsonl.exists():
        _run_tracker_jsonl(MONTAGE, post_jsonl, disable_guard=False)
    if not pre_jsonl.exists():
        _run_tracker_jsonl(MONTAGE, pre_jsonl, disable_guard=True)

    cohort, _, reg_valid_pre, total = _replay_cohort(MONTAGE, disable_guard=True)
    _, _, reg_valid_post, _ = _replay_cohort(MONTAGE, disable_guard=False)

    per_frame = raw_counts(str(MONTAGE), CONF, None)
    post_by = load_jsonl(post_jsonl)
    pre_by = load_jsonl(pre_jsonl)

    flow_post = _flow_for_cohort(cohort, post_by, per_frame)
    flow_pre = _flow_for_cohort(cohort, pre_by, per_frame)

    post_buckets = merge_pipeline(per_frame, post_by)
    pre_buckets = merge_pipeline(per_frame, pre_by)

    zero_det_post = len(post_by)
    zero_det_pre = len(pre_by)
    hud_post = post_buckets.get("HUD field-band ate all", 0)
    hud_pre = pre_buckets.get("HUD field-band ate all", 0)

    fixed_in_c_post = _hud_fixed_band(post_by, cohort)
    fixed_in_hud_post = _hud_fixed_band(post_by, list(post_by.keys()))

    cont_valid_post, cont_total = 0, 0
    if CONTINUOUS.exists():
        _, _, cont_valid_post, cont_total = _replay_cohort(CONTINUOUS, disable_guard=False)

    verdict_lines = []
    tracked = flow_post.get("tracked_ok", 0)
    hud_c = flow_post.get("HUD_ATE_ALL", 0)
    id_sparse = flow_post.get("ID_NONE_sparse", 0) + flow_post.get("ID_NONE_raw0", 0)
    if tracked > hud_c and fixed_in_c_post < hud_c // 2:
        verdict_lines.append(
            "Headline delta likely mixes real recovery (tracked_ok) with id=None overlap; "
            "fixed-band HUD is not the dominant fate of cohort C."
        )
    elif fixed_in_c_post >= hud_c // 2 and hud_c > 0:
        verdict_lines.append(
            "Material cohort-C frames hit HUD_ATE_ALL with field_mask=none (fixed 10/15% band) — "
            "part of the HUD drop may be a band trade, not full recovery."
        )
    else:
        verdict_lines.append(
            "Mixed: see C→outcome table; silent projection win on collapsed tail is independent of headline."
        )

    data = {
        "head": head,
        "fc4a5eb_ancestor": guard_ancestor,
        "note_head": "HEAD may include 90f0db5 ByteTrack/conf after fc4a5eb",
        "montage": str(MONTAGE),
        "cohort_C_size": len(cohort),
        "cohort_C_flow_post": dict(flow_post),
        "cohort_C_flow_pre": dict(flow_pre),
        "fixed_band_HUD_in_C_post": fixed_in_c_post,
        "fixed_band_HUD_all_post": fixed_in_hud_post,
        "reg_valid_pct_pre_replay": round(100 * reg_valid_pre / total, 1),
        "reg_valid_pct_post_replay": round(100 * reg_valid_post / total, 1),
        "zero_det_frames_pre": zero_det_pre,
        "zero_det_frames_post": zero_det_post,
        "zero_det_rate_pre": round(zero_det_pre / total, 4),
        "zero_det_rate_post": round(zero_det_post / total, 4),
        "hud_ate_all_pre": hud_pre,
        "hud_ate_all_post": hud_post,
        "pipeline_buckets_pre": dict(pre_buckets),
        "pipeline_buckets_post": dict(post_buckets),
        "continuous_clip": str(CONTINUOUS) if CONTINUOUS.exists() else None,
        "continuous_reg_valid_pct_post": round(100 * cont_valid_post / cont_total, 1)
        if cont_total
        else None,
        "verdict": verdict_lines,
    }
    OUT_JSON.write_text(json.dumps(data, indent=2))

    lines = [
        "# Phase 0 reconciliation",
        "",
        f"- **HEAD:** `{head}`",
        f"- **fc4a5eb ancestor:** {guard_ancestor}",
        "- **Caveat:** HEAD may include `90f0db5` (ByteTrack/conf); guard-only attribution needs guard-off vs guard-on at same HEAD.",
        "",
        "## Cohort C (pre-guard semantics: valid + area_frac < 0.08)",
        "",
        f"|C| = **{len(cohort)}**",
        "",
        "### C → post-guard pipeline outcome",
        "",
        "| Outcome | Count |",
        "|---------|------:|",
    ]
    for k, v in sorted(flow_post.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            f"- **Fixed-band HUD in C** (`field_mask=none`): **{fixed_in_c_post}**",
            f"- **Fixed-band among all post HUD** ({hud_post} total): **{fixed_in_hud_post}**",
            "",
            "### C → pre-guard (guard disabled) outcome",
            "",
            "| Outcome | Count |",
            "|---------|------:|",
        ]
    )
    for k, v in sorted(flow_pre.items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} |")
    lines.extend(
        [
            "",
            "## Headline / buckets (1minclip)",
            "",
            f"| Metric | Pre-guard tracker | Post-guard tracker |",
            f"|--------|------------------:|-------------------:|",
            f"| zero-det frames | {zero_det_pre} | {zero_det_post} |",
            f"| zero-det rate | {100*zero_det_pre/total:.1f}% | {100*zero_det_post/total:.1f}% |",
            f"| HUD_ATE_ALL | {hud_pre} | {hud_post} |",
            f"| reg_valid (replay) | {100*reg_valid_pre/total:.1f}% | {100*reg_valid_post/total:.1f}% |",
            "",
            "### Pipeline buckets (post)",
            "",
        ]
    )
    for k, v in post_buckets.most_common():
        lines.append(f"- {k}: {v}")
    if cont_total:
        lines.extend(
            [
                "",
                "## Continuous clip gate (`arz_nyj_1min.mov`, 1 min)",
                "",
                f"- post-guard registration_valid: **{100*cont_valid_post/cont_total:.1f}%** ({cont_valid_post}/{cont_total})",
                "- Thin sample; not full-game.",
            ]
        )
    lines.extend(["", "## Verdict", ""] + [f"- {v}" for v in verdict_lines])
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(OUT_MD.read_text())


if __name__ == "__main__":
    main()
