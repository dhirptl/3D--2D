"""Calibration event logging for team classifier."""

from dataclasses import dataclass, field


@dataclass
class CalibrationLog:
    events: list[str] = field(default_factory=list)
    last_distance: float | None = None
    last_fail_reason: str | None = None
    flash_message: str | None = None
    flash_frames_left: int = 0

    def on_calibrate_start(self, n_samples: int) -> None:
        self.events.append(f"CALIBRATING with {n_samples} samples")

    def on_calibrate_success(self, dist: float, c0_size: int, c1_size: int, garbage_size: int) -> None:
        msg = f"LOCKED: centroid_dist={dist:.3f} clusters=({c0_size},{c1_size}) garbage={garbage_size}"
        self.events.append(msg)
        self.last_distance = dist

    def on_calibrate_fail(self, dist: float, threshold: float, attempt: int) -> None:
        msg = f"CALIBRATION FAILED: distance {dist:.3f} < {threshold} (attempt {attempt})"
        self.events.append(msg)
        self.last_distance = dist
        self.last_fail_reason = msg
        self.flash_message = msg
        self.flash_frames_left = 60

    def tick_flash(self) -> str | None:
        if self.flash_frames_left <= 0:
            return None
        self.flash_frames_left -= 1
        return self.flash_message
