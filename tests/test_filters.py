"""ROI/geometry/appearance gates, the persistence tracker, and the sliding window."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import draw_subtitle, gradient_background
from dsf.config import FilterConfig, parse_roi
from dsf.detect.base import Detection, bbox_to_poly, iou, merge_detections
from dsf.filters import (
    GeometryFilter, appearance_ok, persistence_hits, persistence_ok, sliding_window,
)

W, H = 640, 360


def det(x0, y0, x1, y1, score=1.0):
    return Detection(poly=bbox_to_poly(x0, y0, x1, y1), score=score)


# --------------------------------------------------------------------------- ROI

@pytest.mark.parametrize("spec,expected", [
    ("full", (0.0, 0.0, 1.0, 1.0)),
    ("bottom:0.25", (0.0, 0.75, 1.0, 1.0)),
    ("top:0.2", (0.0, 0.0, 1.0, 0.2)),
    ("0.1,0.2,0.9,0.8", (0.1, 0.2, 0.9, 0.8)),
])
def test_parse_roi(spec, expected):
    assert parse_roi(spec) == pytest.approx(expected)


@pytest.mark.parametrize("spec", ["bottom:0", "bottom:1.5", "sideways:0.3", "0.5,0.5,0.1,0.9",
                                  "1,2,3"])
def test_parse_roi_rejects_nonsense(spec):
    with pytest.raises(ValueError):
        parse_roi(spec)


# --------------------------------------------------------------------------- geometry

def test_roi_keeps_subtitles_and_drops_text_higher_up():
    f = GeometryFilter(FilterConfig(roi="bottom:0.30"), W, H)
    subtitle = det(100, 300, 540, 330)
    sign = det(100, 40, 300, 70)
    assert f.keep(subtitle)
    assert not f.keep(sign)


def test_scene_text_mask_widens_the_roi_to_the_whole_frame():
    f = GeometryFilter(FilterConfig(roi="bottom:0.30", scene_text="mask"), W, H)
    assert f.keep(det(100, 40, 300, 70))


def test_size_and_aspect_gates():
    cfg = FilterConfig(roi="full", min_text_height=0.02, max_text_height=0.25,
                       max_aspect=20.0)
    f = GeometryFilter(cfg, W, H)
    assert f.keep(det(10, 100, 200, 130))          # 30px tall = 8.3%
    assert not f.keep(det(10, 100, 200, 103))      # 3px tall, too small
    assert not f.keep(det(10, 10, 200, 340))       # 330px tall, too big
    assert not f.keep(det(0, 100, 640, 120))       # aspect 32:1, too wide


def test_filter_call_returns_only_survivors():
    f = GeometryFilter(FilterConfig(roi="bottom:0.30"), W, H)
    kept = f([det(100, 300, 540, 330), det(100, 40, 300, 70)])
    assert len(kept) == 1


# --------------------------------------------------------------------------- appearance

def test_appearance_accepts_a_real_subtitle():
    frame, _ = draw_subtitle(gradient_background(W, H), "HELLO WORLD", font_size=36)
    assert appearance_ok(frame, det(120, 270, 520, 320), FilterConfig())


def test_appearance_rejects_a_low_contrast_region():
    frame = gradient_background(W, H)
    assert not appearance_ok(frame, det(200, 150, 400, 200), FilterConfig())


def test_appearance_rejects_multicoloured_text():
    """A rainbow sign is scenery; burned-in subtitles are one flat colour."""
    frame = np.full((H, W, 3), 20, dtype=np.uint8)
    region = frame[280:320, 150:450]
    rng = np.random.default_rng(0)
    region[:] = rng.integers(180, 256, region.shape, dtype=np.uint8)
    assert not appearance_ok(frame, det(150, 280, 450, 320),
                             FilterConfig(max_chroma_std=0.02))


# --------------------------------------------------------------------------- persistence

def test_persistence_counts_matching_boxes_across_the_window():
    target = det(100, 300, 500, 330)
    window = [[det(101, 301, 501, 331)], [det(100, 300, 500, 330)], [], [det(99, 299, 499, 329)]]
    assert persistence_hits(target, window, FilterConfig()) == 3


def test_a_one_frame_flash_is_rejected():
    cfg = FilterConfig(min_persist_frames=3)
    flash = det(100, 300, 500, 330)
    window = [[], [], [flash], [], []]
    assert not persistence_ok(flash, window, cfg)


def test_steady_text_is_accepted():
    cfg = FilterConfig(min_persist_frames=3)
    steady = det(100, 300, 500, 330)
    window = [[steady], [steady], [steady], [steady]]
    assert persistence_ok(steady, window, cfg)


def test_short_windows_at_the_clip_edge_are_not_penalised():
    cfg = FilterConfig(min_persist_frames=5)
    d = det(100, 300, 500, 330)
    assert persistence_ok(d, [[d], [d]], cfg)


def test_persistence_is_skipped_when_masking_all_text():
    cfg = FilterConfig(scene_text="mask", min_persist_frames=5)
    assert persistence_ok(det(1, 1, 10, 10), [[]], cfg)


def test_scrolling_credits_survive_when_vertical_scroll_is_allowed():
    """Credits move every frame, so IoU alone would vote them away."""
    cfg = FilterConfig(min_persist_frames=3, allow_vertical_scroll=True)
    frames = [[det(100, 300 - i * 25, 500, 330 - i * 25)] for i in range(5)]
    assert persistence_ok(frames[2][0], frames, cfg)

    strict = FilterConfig(min_persist_frames=3, allow_vertical_scroll=False)
    assert not persistence_ok(frames[2][0], frames, strict)


# --------------------------------------------------------------------------- windowing

def test_sliding_window_emits_every_item_once():
    items = list(range(7))
    centres = [c for c, _ in sliding_window(items, radius=2)]
    assert centres == items


def test_sliding_window_always_contains_its_centre():
    for radius in (0, 1, 3):
        for n in (1, 2, 5, 9):
            for centre, window in sliding_window(range(n), radius):
                assert centre in window
                assert len(window) <= 2 * radius + 1


def test_sliding_window_is_lazy():
    """It must never pull more than the window ahead - clips do not fit in RAM."""
    peak = 0

    def counter():
        nonlocal peak
        for i in range(100):
            peak = i
            yield i

    gen = sliding_window(counter(), radius=2)
    next(gen)
    assert peak <= 2, f"read {peak + 1} items ahead of the first emit"


def test_sliding_window_handles_empty_input():
    assert list(sliding_window([], radius=2)) == []


# --------------------------------------------------------------------------- merging

def test_iou_and_merge():
    a, b = det(0, 0, 100, 100), det(0, 0, 100, 100)
    assert iou(a, b) == pytest.approx(1.0)
    assert iou(a, det(200, 200, 300, 300)) == 0.0

    merged = merge_detections([[a], [b, det(200, 200, 300, 300)]])
    assert len(merged) == 2, "duplicates across detectors collapse, distinct ones survive"
