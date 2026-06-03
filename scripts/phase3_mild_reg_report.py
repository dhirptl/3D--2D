#!/usr/bin/env python3
"""Phase 3: count mild-registration frames (area_frac 0.12-0.30) still valid post-guard."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import FIELD_REG_UPDATE_STRIDE_STABLE, FIELD_REG_STABLE_STREAK
from src.field_registration import FieldRegistration, _projection_area_frac

OUT = ROOT / "outputs" / "phase3_mild_reg_notes.md"


def replay(source: Path) -> tuple[int, int, int]:
    reg = FieldRegistration()
    cap = cv2.VideoCapture(str(source))
    mild = collapsed = proper = 0
    fi = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        fh, fw = frame.shape[:2]
        streak = reg.valid_streak
        stride = FIELD_REG_UPDATE_STRIDE_STABLE if streak >= FIELD_REG_STABLE_STREAK else 1
        if fi % stride == 0:
            reg.update(frame)
        if reg.registration_valid and reg.homography is not None:
            frac = _projection_area_frac(reg.homography, fh, fw)
            if frac is not None:
                if frac < 0.08:
                    collapsed += 1
                elif frac < 0.30:
                    mild += 1
                else:
                    proper += 1
        fi += 1
    cap.release()
    return mild, collapsed, proper


def main() -> None:
    mild, collapsed, proper = replay(ROOT / "1minclip.mov")
    lines = [
        "# Phase 3: mild registration tail (post-guard)",
        "",
        "Collapsed homographies (area_frac < 0.08) are rejected by `fc4a5eb`.",
        "Frames below are **still registration_valid** and may project with imperfect geometry.",
        "",
        "## 1minclip.mov replay (post-guard code)",
        "",
        f"| Band | Valid frames |",
        f"|------|-------------:|",
        f"| area_frac < 0.08 (should be ~0 stored) | {collapsed} |",
        f"| 0.12 <= area_frac < 0.30 (mild) | {mild} |",
        f"| area_frac >= 0.30 (proper) | {proper} |",
        "",
        "## Matcher",
        "",
        "`_yard_match_is_degenerate()` now rejects collinear yard-line matches before `findHomography`.",
        "",
        "## Degradation (continuous clip)",
        "",
        "See `outputs/phase0_reconciliation.md`: `arz_nyj_1min.mov` post-guard valid **66.1%**.",
        "Guard does not appear to collapse valid% toward 40%; no hold-last-good implementation in this phase.",
    ]
    OUT.write_text("\n".join(lines) + "\n")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
