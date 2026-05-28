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
    """Overlap frame decode and encode with inference. Returns frames written.

    Three threads run concurrently:
      reader  → decode frames into read queue
      main    → inference (process_fn), pushes results to write queue
      writer  → encode results to disk (write_fn) from write queue
    """
    read_q: queue.Queue = queue.Queue(maxsize=queue_size)
    write_q: queue.Queue = queue.Queue(maxsize=queue_size)
    stop = threading.Event()

    def reader() -> None:
        try:
            idx = 0
            while cap.isOpened() and not stop.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                read_q.put((idx, frame))
                idx += 1
        finally:
            read_q.put(None)

    def writer() -> None:
        while True:
            item = write_q.get()
            if item is None:
                break
            write_fn(item)

    reader_t = threading.Thread(target=reader, daemon=True)
    writer_t = threading.Thread(target=writer, daemon=True)
    reader_t.start()
    writer_t.start()

    written = 0
    try:
        while True:
            item = read_q.get(timeout=30)
            if item is None:
                break
            fidx, frame = item
            result = process_fn(frame, fidx)
            write_q.put(result)
            written += 1
    except Exception:
        stop.set()
        raise
    finally:
        write_q.put(None)  # signal writer to exit
        writer_t.join(timeout=10)
        reader_t.join(timeout=5)
    return written
