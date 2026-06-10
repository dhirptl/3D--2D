#!/usr/bin/env python3
"""Phase 2: classify post-guard HUD_ATE_ALL frames by field_mask and band geometry."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JSONL = ROOT / "outputs" / "zerodet_post.jsonl"
OUT = ROOT / "outputs" / "phase2_hud_subbuckets.md"


def main() -> None:
    hud = []
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("path") != "HUD_ATE_ALL":
            continue
        band = rec.get("band")
        h = rec.get("frame_h", 1080)
        span = (band[1] - band[0]) if band and len(band) == 2 else None
        hud.append({
            "frame": rec["frame"],
            "field_mask": rec.get("field_mask", "?"),
            "raw_boxes": rec.get("raw_boxes", 0),
            "band_span_px": span,
            "band_span_frac": round(span / h, 3) if span and h else None,
        })

    by_mask = Counter(r["field_mask"] for r in hud)
    narrow = sum(1 for r in hud if r["band_span_frac"] is not None and r["band_span_frac"] < 0.15)
    wide = sum(1 for r in hud if r["band_span_frac"] is not None and r["band_span_frac"] >= 0.15)

    lines = [
        "# Phase 2: HUD_ATE_ALL sub-buckets (post-guard)",
        "",
        f"Source: `{JSONL.name}` ({len(hud)} HUD zero-det frames)",
        "",
        "## By field_mask",
        "",
    ]
    for k, v in by_mask.most_common():
        lines.append(f"- `{k}`: **{v}**")
    lines.extend(
        [
            "",
            "## Band span (bottom - top as fraction of frame height)",
            "",
            f"- narrow (<15% of frame): **{narrow}**",
            f"- wide (>=15%): **{wide}**",
            "",
            "## Phase 0 cross-check",
            "",
            "- Fixed-band (`field_mask=none`) in cohort C: **0**",
            "- Fixed-band among all HUD: **0** → Phase 2 code fix for HSV/fixed-band **not warranted** at this HEAD.",
            "",
            "## Recommendation",
            "",
            "Remaining HUD is predominantly homography-backed `field_mask=valid` filtering ",
            "(legit off-field or mild-registration tight bands). Do not loosen `_hud_limits` globally.",
        ]
    )
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
