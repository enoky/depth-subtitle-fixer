"""Separate overlaid text (subtitles, credits) from text that was filmed in the scene.

A shop sign or a licence plate has genuine depth we must not destroy. Burned-in text does
not: it is pasted on after the fact, so it sits in a predictable part of the frame, has
flat colour and hard contrast, and holds still for many frames. Every gate here encodes one
of those differences.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Iterator, Sequence

import numpy as np

from .config import FilterConfig, parse_roi
from .detect.base import Detection, iou


def sliding_window(iterable: Iterable, radius: int) -> Iterator[tuple]:
    """Yield ``(item, window)`` where *window* is up to ``2*radius+1`` items around *item*.

    Streams: at most ``2*radius+1`` items are held at once, and the first/last items get a
    truncated window rather than being dropped.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    it = iter(iterable)
    buf: deque = deque(maxlen=2 * radius + 1)
    consumed = 0
    for _ in range(radius + 1):
        try:
            buf.append(next(it))
            consumed += 1
        except StopIteration:
            break
    emitted = 0
    while emitted < consumed:
        offset = consumed - len(buf)  # global index of buf[0]
        yield buf[emitted - offset], list(buf)
        emitted += 1
        try:
            buf.append(next(it))
            consumed += 1
        except StopIteration:
            pass


def luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.709 luma in [0, 1] from a uint8 RGB image."""
    a = rgb.astype(np.float32)
    return (0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]) / 255.0


class GeometryFilter:
    """Region-of-interest, text size and aspect ratio gates."""

    def __init__(self, cfg: FilterConfig, width: int, height: int):
        self.cfg = cfg
        self.width, self.height = width, height
        x0, y0, x1, y1 = parse_roi(cfg.roi if cfg.scene_text == "keep" else "full")
        self.roi = (x0 * width, y0 * height, x1 * width, y1 * height)

    def _roi_overlap(self, det: Detection) -> float:
        bx0, by0, bx1, by1 = det.bbox
        rx0, ry0, rx1, ry1 = self.roi
        iw = max(0.0, min(bx1, rx1) - max(bx0, rx0))
        ih = max(0.0, min(by1, ry1) - max(by0, ry0))
        area = max(1e-6, (bx1 - bx0) * (by1 - by0))
        return (iw * ih) / area

    def keep(self, det: Detection) -> bool:
        h, w = det.height, det.width
        if h <= 1 or w <= 1:
            return False
        rel_h = h / self.height
        if not (self.cfg.min_text_height <= rel_h <= self.cfg.max_text_height):
            return False
        if w / h > self.cfg.max_aspect:
            return False
        return self._roi_overlap(det) >= 0.5

    def __call__(self, dets: Sequence[Detection]) -> list[Detection]:
        return [d for d in dets if self.keep(d)]


def appearance_ok(frame: np.ndarray, det: Detection, cfg: FilterConfig) -> bool:
    """Overlaid text is high contrast and flat coloured; scene text usually is not."""
    x0, y0, x1, y1 = det.bbox
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    crop = frame[y0:y1, x0:x1]
    lum = luminance(crop)
    p50, p90, p95 = np.percentile(lum, (50, 90, 95))
    if float(p95 - p50) < cfg.min_contrast:
        return False

    # Colour flatness, measured only on the brightest (candidate glyph) pixels.
    bright = crop[lum >= p90].astype(np.float32)
    if bright.size == 0:
        return False
    total = bright.sum(axis=1, keepdims=True)
    total[total < 1.0] = 1.0
    chroma = bright[:, :2] / total  # normalised r, g
    if float(chroma.std(axis=0).mean()) > cfg.max_chroma_std:
        return False
    return True


def _tracks_match(a: Detection, b: Detection, cfg: FilterConfig) -> bool:
    if iou(a, b) >= cfg.persist_iou:
        return True
    if not cfg.allow_vertical_scroll:
        return False
    # Scrolling credits: same glyph run, translated vertically between frames.
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    aw, ah = ax1 - ax0, ay1 - ay0
    bw, bh = bx1 - bx0, by1 - by0
    if ah <= 0 or bh <= 0 or aw <= 0 or bw <= 0:
        return False
    if abs(aw - bw) > 0.15 * max(aw, bw) or abs(ah - bh) > 0.20 * max(ah, bh):
        return False
    overlap_x = max(0, min(ax1, bx1) - max(ax0, bx0))
    if overlap_x < 0.7 * min(aw, bw):
        return False
    return abs(ay0 - by0) <= 2.5 * max(ah, bh)


def persistence_hits(det: Detection, window: Sequence[Sequence[Detection]],
                     cfg: FilterConfig) -> int:
    """How many frames in the window contain a detection matching *det*."""
    return sum(1 for frame_dets in window
               if any(_tracks_match(det, other, cfg) for other in frame_dets))


def persistence_ok(det: Detection, window: Sequence[Sequence[Detection]],
                   cfg: FilterConfig) -> bool:
    if cfg.scene_text != "keep" or cfg.min_persist_frames <= 1:
        return True
    # A short window near the clip edges must not be penalised for being short.
    required = min(cfg.min_persist_frames, len(window))
    return persistence_hits(det, window, cfg) >= required
