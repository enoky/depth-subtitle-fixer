"""Text detectors. Each backend is imported lazily so a missing optional model
never breaks the rest of the pipeline."""

from __future__ import annotations

from .base import Detection, DetectorResult, TextDetector, resolve_device

__all__ = ["Detection", "DetectorResult", "TextDetector", "resolve_device", "build_detectors"]


def build_detectors(names, cfg, device: str | None = None) -> list[TextDetector]:
    """Instantiate the requested detectors, skipping (with a warning) any that fail to load."""
    import warnings

    device = resolve_device(device or cfg.device)
    built: list[TextDetector] = []
    for name in names:
        name = name.strip().lower()
        if not name:
            continue
        try:
            if name == "doctr":
                from .doctr_det import DoctrDetector

                built.append(DoctrDetector(cfg, device))
            elif name == "easyocr":
                from .easyocr_det import EasyOcrDetector

                built.append(EasyOcrDetector(cfg, device))
            else:
                raise ValueError(f"unknown detector {name!r} (choose from: doctr, easyocr)")
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort
            if len(names) == 1:
                raise
            warnings.warn(f"detector {name!r} unavailable, continuing without it: {exc}")
    if not built:
        raise RuntimeError("no text detector could be loaded")
    return built
