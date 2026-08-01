"""Temporal smoothing of the alpha-mask stack.

Frame-by-frame detection flickers: a subtitle that is found on frames 100-140 will typically
be missed on two or three of them, which shows up as the text popping in and out of the depth
map. Masks are buffered as uint8 (a quarter the memory of float32) and filtered across a small
window before compositing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import TemporalConfig


def to_u8(alpha: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)


def from_u8(alpha_u8: np.ndarray) -> np.ndarray:
    return alpha_u8.astype(np.float32) / 255.0


def smooth(center: np.ndarray, window: Sequence[np.ndarray], cfg: TemporalConfig) -> np.ndarray:
    """Collapse a window of uint8 alpha masks into one.

    ``median`` rejects isolated false positives *and* fills isolated misses; ``max`` only
    fills misses, which is what scrolling credits want since a moving glyph must never be
    voted away by its neighbours. *center* is passed explicitly because windows are truncated
    at the clip boundaries, so the centre is not always at ``len(window) // 2``.
    """
    if cfg.mode == "none" or len(window) <= 1:
        return center
    stack = np.stack(window, axis=0)
    if cfg.mode == "median":
        return np.median(stack, axis=0).astype(np.uint8)
    if cfg.mode == "max":
        return stack.max(axis=0)
    raise ValueError(f"unknown temporal mode {cfg.mode!r}; use median, max or none")
