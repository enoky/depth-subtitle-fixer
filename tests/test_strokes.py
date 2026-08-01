"""Glyph extraction: the mask must hug the strokes, not the box."""

from __future__ import annotations

import numpy as np

from conftest import draw_subtitle, draw_subtitle_full, gradient_background, text_bbox
from dsf.config import StrokeConfig
from dsf.detect.base import Detection, bbox_to_poly
from dsf.refine.strokes import compose_alpha, extract_patch


def _extract(frame, box, cfg=None):
    det = Detection(poly=bbox_to_poly(*box), score=1.0)
    return extract_patch(frame, det, cfg or StrokeConfig())


def test_extracted_alpha_covers_the_glyphs_and_spares_the_background():
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    assert patch is not None

    alpha = compose_alpha([patch], 360, 640)
    recall = float((alpha[fill > 200] > 0.5).mean())
    # False positives are judged against the whole rendered text - fill plus outline. The
    # outline is part of the burned-in overlay and its depth is corrupted too, so covering
    # it is correct, not a miss.
    box = np.zeros((360, 640), dtype=bool)
    box[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = True
    background = box & (full == 0)
    false_positive = float((alpha[background] > 0.5).mean())

    assert recall > 0.90, f"only {recall:.1%} of glyph pixels recovered"
    assert false_positive < 0.10, f"{false_positive:.1%} of the background leaked in"


def test_mask_reaches_the_outline_not_just_the_stroke_core():
    """The dark rim is part of the overlay; leaving it unmasked leaves bad depth behind."""
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    alpha = compose_alpha([patch], 360, 640)

    rim = (full > 0) & (fill == 0)
    assert float((alpha[rim] > 0.5).mean()) > 0.5, "outline left unmasked"


def test_mask_is_far_smaller_than_the_box():
    """The whole point: stamping the box would flatten a slab of real depth."""
    bg = gradient_background(640, 360)
    frame, truth = draw_subtitle(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(truth)
    patch = _extract(frame, (x0, y0, x1, y1))
    assert patch is not None

    box_area = (x1 - x0) * (y1 - y0)
    covered = float((patch.alpha > 0.5).sum())
    assert covered < 0.5 * box_area


def test_alpha_edges_are_soft():
    """Anti-aliased boundaries must survive - a hard edge rings after stereo warping."""
    bg = gradient_background(640, 360)
    frame, truth = draw_subtitle(bg, "HELLO", font_size=48)
    x0, y0, x1, y1 = text_bbox(truth)
    patch = _extract(frame, (x0 - 4, y0 - 4, x1 + 4, y1 + 4))
    assert patch is not None
    partial = (patch.alpha > 0.05) & (patch.alpha < 0.95)
    assert partial.sum() > 0


def test_dark_text_on_a_light_background():
    bg = np.full((360, 640, 3), 230, dtype=np.uint8)
    frame, truth = draw_subtitle(bg, "DARK TEXT", font_size=40,
                                 fill=(10, 10, 10), outline=(250, 250, 250), stroke=3)
    x0, y0, x1, y1 = text_bbox(truth)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    assert patch is not None
    alpha = compose_alpha([patch], 360, 640)
    assert float((alpha[truth > 200] > 0.5).mean()) > 0.85


def test_flat_background_yields_no_mask():
    """A box containing no text at all must produce nothing, not noise."""
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    assert _extract(frame, (200, 200, 400, 260)) is None


def test_smooth_gradient_is_not_mistaken_for_text():
    frame = gradient_background(640, 360)
    patch = _extract(frame, (200, 150, 460, 220))
    if patch is not None:
        assert float((patch.alpha > 0.5).mean()) < 0.05


def test_outline_check_rejects_glyphs_without_a_dark_rim():
    """Text with no outline and barely any rim contrast looks like scenery."""
    bg = np.full((360, 640, 3), 245, dtype=np.uint8)
    frame, truth = draw_subtitle(bg, "NO OUTLINE", font_size=40,
                                 fill=(255, 255, 255), outline=(255, 255, 255), stroke=0)
    x0, y0, x1, y1 = text_bbox(truth)
    box = (x0 - 6, y0 - 6, x1 + 6, y1 + 6)

    strict = _extract(frame, box, StrokeConfig(outline_check=True))
    relaxed = _extract(frame, box, StrokeConfig(outline_check=False))
    strict_area = 0 if strict is None else int((strict.alpha > 0.5).sum())
    relaxed_area = 0 if relaxed is None else int((relaxed.alpha > 0.5).sum())
    assert relaxed_area > strict_area


def test_compose_alpha_merges_patches_with_max():
    from dsf.refine.strokes import AlphaPatch

    det = Detection(poly=bbox_to_poly(0, 0, 4, 4))
    a = AlphaPatch(x0=0, y0=0, alpha=np.full((4, 4), 0.4, np.float32), det=det)
    b = AlphaPatch(x0=2, y0=2, alpha=np.full((4, 4), 0.9, np.float32), det=det)
    out = compose_alpha([a, b], 8, 8)
    assert out[0, 0] == np.float32(0.4)
    assert out[3, 3] == np.float32(0.9)  # overlap takes the stronger alpha
    assert out[5, 5] == np.float32(0.9)
    assert out[7, 7] == 0.0


def test_patches_clip_at_the_frame_edge():
    from dsf.refine.strokes import AlphaPatch

    det = Detection(poly=bbox_to_poly(0, 0, 4, 4))
    patch = AlphaPatch(x0=6, y0=6, alpha=np.ones((4, 4), np.float32), det=det)
    out = compose_alpha([patch], 8, 8)
    assert out.shape == (8, 8)
    assert out[7, 7] == 1.0
