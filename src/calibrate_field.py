"""Manual 4-point homography calibration for a football clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.config import FIELD_REG_TEMPLATE_H, FIELD_REG_TEMPLATE_W


DEFAULT_FIELD_PTS = np.array(
    [
        [0.0, 0.0],
        [float(FIELD_REG_TEMPLATE_W), 0.0],
        [float(FIELD_REG_TEMPLATE_W), float(FIELD_REG_TEMPLATE_H)],
        [0.0, float(FIELD_REG_TEMPLATE_H)],
    ],
    dtype=np.float64,
)


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {frame_idx} from: {video_path}")
    return frame


def calibrate(video_path: Path, out_json: Path, frame_idx: int = 0) -> np.ndarray:
    frame = _load_frame(video_path, frame_idx)
    draw = frame.copy()
    clicks: list[list[int]] = []
    win = "click 4 field landmarks (clockwise)"

    def on_click(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 4:
            clicks.append([int(x), int(y)])
            cv2.circle(draw, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(
                draw,
                str(len(clicks)),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.imshow(win, draw)

    cv2.imshow(win, draw)
    cv2.setMouseCallback(win, on_click)
    while len(clicks) < 4:
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
    if len(clicks) != 4:
        raise RuntimeError("Calibration aborted: exactly 4 clicks required.")

    img_pts = np.array(clicks, dtype=np.float64)
    h, _ = cv2.findHomography(img_pts, DEFAULT_FIELD_PTS)
    if h is None:
        raise RuntimeError("findHomography failed.")

    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": str(video_path),
        "frame": int(frame_idx),
        "H": h.tolist(),
        "img_pts": clicks,
        "field_pts": DEFAULT_FIELD_PTS.tolist(),
        "template_w": int(FIELD_REG_TEMPLATE_W),
        "template_h": int(FIELD_REG_TEMPLATE_H),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    return h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Video clip path")
    parser.add_argument("--out", required=True, help="Output calibration JSON")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to calibrate on")
    args = parser.parse_args()
    h = calibrate(Path(args.source), Path(args.out), frame_idx=args.frame)
    print(f"Saved calibration to {args.out}")
    print("Homography:")
    print(h)


if __name__ == "__main__":
    main()
