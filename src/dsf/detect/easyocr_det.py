"""EasyOCR / CRAFT text detection.

CRAFT is character-region based and trained on scene text, which makes it noticeably better
than a document-leaning detector on stylised intro credits. We only ever load the detector -
the recognition model is skipped, so no recogniser weights are downloaded.
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from .base import Detection, DetectorResult, bbox_to_poly


class EasyOcrDetector:
    name = "easyocr"

    def __init__(self, cfg, device: str, languages: Sequence[str] = ("en",)):
        import easyocr

        self.cfg = cfg
        storage = os.environ.get("EASYOCR_MODULE_PATH")
        self.reader = easyocr.Reader(
            list(languages),
            gpu=device.startswith("cuda"),
            detector=True,
            recognizer=False,
            verbose=False,
            model_storage_directory=storage,
            download_enabled=True,
        )

    def detect(self, frames: Sequence[np.ndarray]) -> list[DetectorResult]:
        results: list[DetectorResult] = []
        for frame in frames:
            h, w = frame.shape[:2]
            dets: list[Detection] = []
            try:
                horizontal, free = self.reader.detect(
                    np.ascontiguousarray(frame),
                    canvas_size=max(w, h),
                    mag_ratio=1.0,
                )
            except Exception:
                results.append(DetectorResult())
                continue

            for box in _unwrap(horizontal):
                # EasyOCR horizontal boxes are [x_min, x_max, y_min, y_max]
                if len(box) < 4:
                    continue
                x0, x1, y0, y1 = (float(v) for v in box[:4])
                if x1 <= x0 or y1 <= y0:
                    continue
                dets.append(Detection(poly=bbox_to_poly(x0, y0, x1, y1),
                                      score=1.0, source="easyocr"))
            for poly in _unwrap(free):
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                if len(pts) >= 3:
                    dets.append(Detection(poly=pts, score=1.0, source="easyocr"))

            dets = [d.clipped(w, h) for d in dets]
            results.append(DetectorResult(detections=dets))
        return results


def _unwrap(value) -> list:
    """EasyOCR wraps per-image results in an extra list; flatten exactly one level."""
    if not value:
        return []
    first = value[0]
    if isinstance(first, (list, tuple, np.ndarray)) and len(first) and \
            isinstance(first[0], (list, tuple, np.ndarray)):
        return list(first)
    return list(value)
