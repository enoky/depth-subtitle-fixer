"""Common detector interface.

A detector consumes RGB frames and returns polygons in absolute pixel coordinates, plus an
optional soft probability map used as a region prior during stroke extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


@dataclass
class Detection:
    """One detected text region, in absolute pixel coordinates."""

    poly: np.ndarray  # (N, 2) float32
    score: float = 1.0
    source: str = ""
    #: Cached `bbox`. Not an argument and not part of equality - it is derived from `poly`.
    _bbox: tuple[int, int, int, int] | None = field(default=None, init=False, repr=False,
                                                    compare=False)

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Axis-aligned integer (x0, y0, x1, y1), x1/y1 exclusive.

        Computed once and kept. The persistence gate asks every detection in a frame about
        every detection in the window either side of it, so on a credit roll this was called
        784 times per frame for the couple of dozen distinct answers it has - four numpy
        reductions and two array allocations each, and 7% of a scanned frame once the mask
        chain had moved to the GPU and stopped hiding it.

        Safe to cache because `poly` is never written after construction: `clipped` builds a
        new Detection rather than editing this one, and nothing else touches it.
        """
        if self._bbox is None:
            x0, y0 = np.floor(self.poly.min(axis=0)).astype(int)
            x1, y1 = np.ceil(self.poly.max(axis=0)).astype(int)
            self._bbox = (int(x0), int(y0), int(x1), int(y1))
        return self._bbox

    @property
    def width(self) -> int:
        x0, _, x1, _ = self.bbox
        return max(0, x1 - x0)

    @property
    def height(self) -> int:
        _, y0, _, y1 = self.bbox
        return max(0, y1 - y0)

    def clipped(self, width: int, height: int) -> "Detection":
        poly = self.poly.copy()
        np.clip(poly[:, 0], 0, width, out=poly[:, 0])
        np.clip(poly[:, 1], 0, height, out=poly[:, 1])
        return Detection(poly=poly, score=self.score, source=self.source)


@dataclass
class DetectorResult:
    """Per-frame detector output: where the text is, in absolute pixel coordinates.

    Deliberately boxes only. Segmentation detectors also expose a probability map, but
    DBNet-style maps cover a *shrunk* text region and are far coarser than a glyph, so
    folding one into the per-pixel alpha eats the strokes it is meant to confirm.
    Localisation belongs to the detector; per-pixel decisions belong to stroke extraction.
    """

    detections: list[Detection] = field(default_factory=list)


class TextDetector(Protocol):
    name: str

    def detect(self, frames: Sequence[np.ndarray]) -> list[DetectorResult]:
        """Detect text in a batch of uint8 HxWx3 RGB frames."""
        ...


def resolve_device(device: str) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def bbox_to_poly(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)


def iou(a: Detection, b: Detection) -> float:
    ax0, ay0, ax1, ay1 = a.bbox
    bx0, by0, bx1, by1 = b.bbox
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def merge_detections(groups: Sequence[Sequence[Detection]], iou_thresh: float = 0.6
                     ) -> list[Detection]:
    """Union detections from several detectors, dropping near-duplicates."""
    merged: list[Detection] = []
    for group in groups:
        for det in group:
            for kept in merged:
                if iou(det, kept) >= iou_thresh:
                    kept.score = max(kept.score, det.score)
                    break
            else:
                merged.append(det)
    return merged
