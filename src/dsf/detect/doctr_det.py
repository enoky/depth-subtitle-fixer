"""docTR (DBNet / FAST / LinkNet) text detection.

Segmentation-based, PyTorch-native, so it runs on the same CUDA build we install for
everything else. Boxes only - see DetectorResult for why the probability map is not used.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import Detection, DetectorResult, bbox_to_poly

ARCHS = (
    "db_resnet50", "db_resnet34", "db_mobilenet_v3_large",
    "fast_base", "fast_small", "fast_tiny",
    "linknet_resnet50", "linknet_resnet34", "linknet_resnet18",
)


class DoctrDetector:
    name = "doctr"

    def __init__(self, cfg, device: str):
        from doctr.models import detection_predictor

        self.cfg = cfg
        self.device = device
        self.predictor = detection_predictor(
            arch=cfg.det_arch,
            pretrained=True,
            assume_straight_pages=True,
            preserve_aspect_ratio=True,
            symmetric_pad=True,
            batch_size=cfg.batch_size,
        )
        self.predictor.model = self.predictor.model.to(device).eval()

    @staticmethod
    def _to_detections(arr: np.ndarray, w: int, h: int, min_score: float) -> list[Detection]:
        dets: list[Detection] = []
        if arr is None or len(arr) == 0:
            return dets
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2:  # straight pages: (N, 5) -> xmin, ymin, xmax, ymax, score
            for row in arr:
                score = float(row[4]) if row.shape[0] > 4 else 1.0
                if score < min_score:
                    continue
                x0, y0, x1, y1 = row[:4]
                dets.append(Detection(
                    poly=bbox_to_poly(x0 * w, y0 * h, x1 * w, y1 * h),
                    score=score, source="doctr",
                ))
        elif arr.ndim == 3:  # rotated pages: (N, P, 2), optionally with a score row
            for poly in arr:
                pts = poly[:, :2].astype(np.float32)
                pts[:, 0] *= w
                pts[:, 1] *= h
                dets.append(Detection(poly=pts, score=1.0, source="doctr"))
        return dets

    def detect(self, frames: Sequence[np.ndarray]) -> list[DetectorResult]:
        pages = [np.ascontiguousarray(f) for f in frames]
        preds = self.predictor(pages)

        results: list[DetectorResult] = []
        for page, pred in zip(pages, preds):
            h, w = page.shape[:2]
            arr = None
            if isinstance(pred, dict):
                arr = pred.get("words")
                if arr is None and pred:
                    arr = next(iter(pred.values()))
            results.append(DetectorResult(
                detections=self._to_detections(arr, w, h, self.cfg.min_score)))
        return results
