"""Optional read-ahead threading for video decode (inference stays on main thread)."""

import queue
import threading

import cv2
import numpy as np


def run_with_read_ahead(
    cap: cv2.VideoCapture,
    process_fn,
    write_fn,
    *,
    queue_size: int = 2,
) -> int:
    """Overlap frame decode with processing. Returns frames written."""
    q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop = threading.Event()

    def reader() -> None:
        try:
            idx = 0
            while cap.isOpened() and not stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                q.put((idx, frame))
                idx += 1
        finally:
            q.put(None)  # always send sentinel so consumer never blocks forever

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    written = 0
    try:
        while True:
            item = q.get(timeout=30)  # guard against reader dying without sentinel
            if item is None:
                break
            fidx, frame = item
            result = process_fn(frame, fidx)
            write_fn(result)
            written += 1
    except Exception:
        stop.set()  # signal reader to exit if process_fn raised
        raise
    finally:
        t.join(timeout=5)
    return written
