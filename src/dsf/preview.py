"""Contact-sheet previews.

Four panels per frame - source, extracted glyph mask, depth before, depth after - which is
the fastest way to see whether the thresholds are right before committing to a full render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .composite import composite_frame, resize_alpha
from .config import PipelineConfig
from .detect.base import Detection
from .temporal import from_u8

PANEL_LABELS = ("source + detections", "glyph mask", "depth (before)", "depth (after)")


def depth_to_display(depth_y: np.ndarray, bit_depth: int) -> np.ndarray:
    """Scale a luma plane to 8-bit for viewing (display only - never fed back to encoding)."""
    max_code = float((1 << bit_depth) - 1)
    return np.clip(depth_y.astype(np.float32) * 255.0 / max_code, 0, 255).astype(np.uint8)


def draw_detections(rgb: np.ndarray, detections: Sequence[Detection]) -> np.ndarray:
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for det in detections:
        pts = det.poly.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], isClosed=True, color=(0, 220, 255), thickness=2)
    return out


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    h = out.shape[0]
    bar = max(24, h // 22)
    cv2.rectangle(out, (0, 0), (out.shape[1], bar), (0, 0, 0), -1)
    scale = bar / 34.0
    cv2.putText(out, text, (8, int(bar * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), max(1, int(scale * 2)), cv2.LINE_AA)
    return out


def contact_sheet(rgb: np.ndarray, alpha: np.ndarray, depth_before: np.ndarray,
                  depth_after: np.ndarray, bit_depth: int,
                  detections: Sequence[Detection] = (), panel_width: int = 640) -> np.ndarray:
    """Build a 2x2 BGR contact sheet."""
    panels = [
        draw_detections(rgb, detections),
        cv2.cvtColor(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(depth_to_display(depth_before, bit_depth), cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(depth_to_display(depth_after, bit_depth), cv2.COLOR_GRAY2BGR),
    ]
    resized = []
    for panel, label in zip(panels, PANEL_LABELS):
        h, w = panel.shape[:2]
        ph = max(1, int(round(panel_width * h / w)))
        small = cv2.resize(panel, (panel_width, ph), interpolation=cv2.INTER_AREA)
        resized.append(_label(small, label))

    target_h = max(p.shape[0] for p in resized)
    padded = [cv2.copyMakeBorder(p, 0, target_h - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20)) for p in resized]
    top = np.hstack(padded[:2])
    bottom = np.hstack(padded[2:])
    return np.vstack([top, bottom])


def write_previews(rgb_path: str, depth_path: str, frame_indices: Sequence[int],
                   cfg: PipelineConfig, out_dir: str | Path,
                   panel_width: int = 640) -> list[Path]:
    """Render and save a contact sheet per requested frame index."""
    from .pipeline import masks_for_frames, sample_depth, sample_frames
    from .videoio import probe

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb_info, depth_info = probe(rgb_path), probe(depth_path)

    frames = sample_frames(rgb_path, frame_indices)
    depths = sample_depth(depth_path, frame_indices, depth_info)
    masks = masks_for_frames(rgb_path, cfg, frame_indices, rgb_info)

    written: list[Path] = []
    for idx in sorted(frames):
        if idx not in depths or idx not in masks:
            continue
        rgb = frames[idx]
        depth_frame = depths[idx]
        mask_u8, detections = masks[idx]
        alpha_rgb = from_u8(mask_u8)
        alpha = resize_alpha(alpha_rgb, depth_info.width, depth_info.height)
        after = composite_frame(depth_frame.y, alpha, cfg.composite,
                                depth_info.bit_depth, depth_info.color_range)
        sheet = contact_sheet(rgb, alpha_rgb, depth_frame.y, after,
                              depth_info.bit_depth, detections=detections,
                              panel_width=panel_width)
        path = out_dir / f"preview_{idx:06d}.png"
        cv2.imwrite(str(path), sheet)
        written.append(path)
    return written
