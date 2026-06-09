"""Stable eval report schema for regression gate (DESIGN §3)."""

from __future__ import annotations

REQUIRED_REPORT_KEYS = frozenset({
    "frames",
    "avg_detections",
    "zero_det_rate",
    "track_id_change_rate",
    "label_flip_rate",
    "locked_pct",
})

DECOMPOSED_KEYS = frozenset({
    "model_miss_rate",
    "id_none_dropped_rate",
    "hud_dropped_rate",
    "field_dropped_rate",
    "mask_dropped_rate",
    "actionable_zero_det_rate",
})

OPTIONAL_TEAM_KEYS = frozenset({
    "team_accuracy",
    "team_per_track_accuracy",
    "cluster_silhouette",
})

BASELINE_METRIC_KEYS = (
    "avg_detections",
    "zero_det_rate",
    "actionable_zero_det_rate",
    "track_id_change_rate",
    "label_flip_rate",
    "model_miss_rate",
    "id_none_dropped_rate",
    "hud_dropped_rate",
)


def validate_report_schema(report: dict) -> list[str]:
    """Return list of missing required keys (empty if valid)."""
    missing = [k for k in REQUIRED_REPORT_KEYS if k not in report]
    return missing


def extract_baseline_metrics(report: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in BASELINE_METRIC_KEYS:
        if key in report and report[key] is not None:
            out[key] = float(report[key])
    return out


def enrich_report_with_decomposed(report: dict, design_rates: dict) -> dict:
    """Merge DESIGN bucket rates into eval report in place."""
    report.update(design_rates)
    return report
