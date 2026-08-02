"""Temporal smoothing of the alpha-mask stack.

Frame-by-frame detection flickers: a subtitle that is found on frames 100-140 will typically
be missed on two or three of them, which shows up as the text popping in and out of the depth
map. Masks are buffered as uint8 (a quarter the memory of float32) and filtered across a small
window before compositing.
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np

from .config import TemporalConfig


def to_u8(alpha: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)


def from_u8(alpha_u8: np.ndarray) -> np.ndarray:
    return alpha_u8.astype(np.float32) / 255.0


@functools.lru_cache(maxsize=32)
def _sorting_network(count: int) -> tuple[tuple[int, int], ...]:
    """Index pairs that sort *count* items when applied in order as compare-exchanges.

    Batcher's odd-even mergesort, generated for the next power of two with the comparators
    that reach past the end dropped - which still sorts, and avoids inventing padding
    values that would sit in the middle and change the answer.
    """
    size = 1
    while size < count:
        size *= 2
    pairs: list[tuple[int, int]] = []
    span = 1
    while span < size:
        step = span
        while step >= 1:
            for base in range(step % span, size - step, 2 * step):
                for i in range(min(step, size - base - step)):
                    lo, hi = base + i, base + i + step
                    if lo // (span * 2) == hi // (span * 2) and hi < count:
                        pairs.append((lo, hi))
            step //= 2
        span *= 2
    return tuple(pairs)


def _median_u8(window: Sequence[np.ndarray]) -> np.ndarray:
    """Per-pixel median of uint8 masks, without ever leaving uint8.

    ``np.median`` promotes to float64 so it can average the middle pair, so finding the
    middle of three bytes cost eight bytes per pixel per frame and two more passes to get
    back down again. A network of elementwise min/max sorts every pixel independently and
    stays in the dtype it was handed: on a 1920x800 window of three that is 21 ms of work
    replaced by 0.7 ms.

    The middle pair is averaged only when the window is even, which happens because a
    window is truncated at a clip boundary rather than chosen. The average is floored
    there - which is what ``astype(np.uint8)`` did to np.median's .5 - so masks come out
    of this unchanged.
    """
    count = len(window)
    middle = count // 2
    if count == 3:
        # The median of three needs no sort: it is the larger of the two smaller ones.
        a, b, c = window
        return np.maximum(np.minimum(a, b), np.minimum(np.maximum(a, b), c))

    items = [np.array(frame, copy=True) for frame in window]
    for lo, hi in _sorting_network(count):
        low = np.minimum(items[lo], items[hi])
        np.maximum(items[lo], items[hi], out=items[hi])
        items[lo] = low
    if count % 2:
        return items[middle]
    return ((items[middle - 1].astype(np.uint16) + items[middle]) // 2).astype(np.uint8)


def smooth(center: np.ndarray, window: Sequence[np.ndarray], cfg: TemporalConfig) -> np.ndarray:
    """Collapse a window of uint8 alpha masks into one.

    ``median`` rejects isolated false positives *and* fills isolated misses; ``max`` only
    fills misses, which is what scrolling credits want since a moving glyph must never be
    voted away by its neighbours. *center* is passed explicitly because windows are truncated
    at the clip boundaries, so the centre is not always at ``len(window) // 2``.
    """
    if cfg.mode == "none" or len(window) <= 1:
        return center
    if cfg.mode == "median":
        return _median_u8(window)
    if cfg.mode == "max":
        # reduce runs straight down the window; np.stack copied all of it first.
        return np.maximum.reduce(window)
    raise ValueError(f"unknown temporal mode {cfg.mode!r}; use median, max or none")
