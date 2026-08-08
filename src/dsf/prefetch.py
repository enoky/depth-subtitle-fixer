"""Run a reader ahead of its consumer, so decoding and inference stop taking turns.

Decoding a frame and detecting text in it use different hardware, and left alone they
alternate: ffmpeg sits idle while the GPU works, and the GPU sits idle while ffmpeg
decodes. Measured on a 1080p clip the GPU was below 20% utilisation for 65% of the wall
clock. Reading one step ahead overlaps them, which is worth about a third of the detection
rate on a continuous read.

The ceiling is the slower of the two stages - this can never do better than removing the
faster one from the bill - and the queue is deliberately shallow, because each item is a
full uncompressed frame.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

#: Items the reader may run ahead. Small on purpose: at 4K a depth of three already holds
#: 75 MB of raw frames.
DEFAULT_DEPTH = 3

#: Raw frame bytes a prefetch queue may hold, used to size a depth against a frame size.
DEFAULT_BUDGET = 256 << 20


def depth_for(width: int, height: int, wanted: int,
              budget: int = DEFAULT_BUDGET, channels: int = 3) -> int:
    """How far ahead we can afford to read at this frame size.

    A depth chosen in frames means something quite different at 720p and at 8K, and the
    point of streaming everything is that clip length is bounded by disk rather than RAM.
    """
    per_frame = max(1, int(width) * int(height) * int(channels))
    return max(1, min(int(wanted), int(budget) // per_frame))


def prefetch(stream: Iterator, depth: int = DEFAULT_DEPTH) -> Iterator:
    """Yield from *stream* with a worker thread running ahead of the consumer.

    The worker owns the stream and closes it, including when the consumer stops early -
    which it does on every clip that has seen enough. Anything the stream raises is
    re-raised on the consumer's thread, so a decode failure cannot vanish into a worker.
    """
    box: queue.Queue = queue.Queue(maxsize=max(1, depth))
    finished = object()
    stop = threading.Event()

    def pump() -> None:
        try:
            for item in stream:
                if stop.is_set():
                    break
                box.put(item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the consumer's thread
            box.put(exc)
        finally:
            close = getattr(stream, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001 - already on the way out
                    pass
            box.put(finished)

    worker = threading.Thread(target=pump, name="dsf-prefetch", daemon=True)
    worker.start()
    drained = False
    try:
        while True:
            item = box.get()
            if item is finished:
                drained = True
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        # Only when the consumer stopped early, which leaves the reader parked on a full
        # queue with ffmpeg still running. Draining lets it see the stop flag and close the
        # stream. Skipping this when the stream already ended matters: a blocking get on an
        # empty queue waits out its whole timeout, and doing that once per cluster turned a
        # 25s scan into a 265s one.
        if not drained:
            deadline = time.monotonic() + 10.0
            while worker.is_alive() and time.monotonic() < deadline:
                try:
                    box.get(timeout=0.1)
                except queue.Empty:
                    pass
        worker.join(timeout=5.0)
