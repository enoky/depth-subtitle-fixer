"""Glyph extraction: the mask must hug the strokes, not the box."""

from __future__ import annotations

import cv2
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
    # Recall is judged on the *interior* of the fill. A pixel on the stroke boundary is part
    # glyph and part outline, so it reads at neither one's level - it is text under any
    # reading, it sits against covered rim, and holding it to a hard 0.5 would be measuring
    # anti-aliasing rather than whether the word got masked.
    interior = cv2.erode((fill > 200).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    recall = float((alpha[interior] > 0.5).mean())
    # False positives are judged against the whole rendered text - fill plus outline. The
    # outline is part of the burned-in overlay and its depth is corrupted too, so covering
    # it is correct, not a miss.
    box = np.zeros((360, 640), dtype=bool)
    box[y0 - 6:y1 + 6, x0 - 6:x1 + 6] = True
    background = box & (full == 0)
    false_positive = float((alpha[background] > 0.5).mean())

    assert recall > 0.90, f"only {recall:.1%} of glyph interior recovered"
    assert false_positive < 0.05, f"{false_positive:.1%} of the background leaked in"


def test_mask_reaches_the_outline_not_just_the_stroke_core():
    """The dark rim is part of the overlay; leaving it unmasked leaves bad depth behind."""
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    alpha = compose_alpha([patch], 360, 640)

    rim = (full > 0) & (fill == 0)
    assert float((alpha[rim] > 0.5).mean()) > 0.5, "outline left unmasked"


def test_mask_spares_the_background_inside_the_box():
    """The whole point: stamping the box would flatten a slab of real scene depth.

    Measured as background actually covered rather than as a fraction of the box. The mask
    deliberately includes each glyph's outline, and for a tight box around dense text the
    rendered text really is most of that box - so a share-of-box budget would be measuring
    how wordy the subtitle is, not whether any real depth was destroyed.
    """
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 8, y0 - 8, x1 + 8, y1 + 8))
    assert patch is not None

    alpha = compose_alpha([patch], 360, 640)
    box = np.zeros((360, 640), dtype=bool)
    box[y0 - 8:y1 + 8, x0 - 8:x1 + 8] = True
    background = box & (full == 0)

    assert float((alpha[background] > 0.5).mean()) < 0.05, "real depth would be flattened"
    assert float((alpha[box] > 0.5).mean()) < 0.80, "mask should not swallow the whole box"


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


def test_text_without_any_outline_is_still_found():
    """Credit rolls are usually drawn straight onto the picture with no rim at all."""
    bg = gradient_background(640, 360)
    frame, fill, _ = draw_subtitle_full(bg, "NO OUTLINE", font_size=40,
                                        fill=(255, 255, 255), outline=(255, 255, 255),
                                        stroke=0)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    assert patch is not None
    alpha = compose_alpha([patch], 360, 640)
    assert float((alpha[fill > 200] > 0.5).mean()) > 0.85


def test_barely_visible_text_is_rejected_as_noise():
    """A crop whose strokes carry almost no contrast is not worth masking."""
    bg = np.full((360, 640, 3), 245, dtype=np.uint8)
    frame, fill, _ = draw_subtitle_full(bg, "INVISIBLE", font_size=40,
                                        fill=(248, 248, 248), outline=(248, 248, 248),
                                        stroke=0)
    x0, y0, x1, y1 = text_bbox(fill)
    assert _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6)) is None


def test_word_over_a_bright_patch_is_not_dropped():
    """The failure this guards against: a detection box straddling a lighting boundary.

    A global luma split spent its classes describing the background instead of separating
    text from it, inverted the polarity, and silently dropped whichever word sat over the
    brighter half - while its neighbour over the dark half came through fine.
    """
    bg = np.full((360, 640, 3), 24, dtype=np.uint8)
    bg[150:290, 40:330] = 128  # a bright shoulder under the left-hand word
    bg = cv2.GaussianBlur(bg, (0, 0), 12)
    frame, fill, _ = draw_subtitle_full(bg, "dianne crittenden", font_size=34, y_frac=0.60,
                                        fill=(255, 255, 255), outline=(255, 255, 255),
                                        stroke=0)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6))
    assert patch is not None
    alpha = compose_alpha([patch], 360, 640)

    columns = np.arange(640)[None, :]
    over_bright = (fill > 200) & (columns < 330)
    over_dark = (fill > 200) & (columns >= 330)
    assert over_bright.sum() > 50 and over_dark.sum() > 50, "test setup: both halves needed"
    bright_recall = float((alpha[over_bright] > 0.5).mean())
    dark_recall = float((alpha[over_dark] > 0.5).mean())
    assert bright_recall > 0.80, f"word over the bright patch was dropped ({bright_recall:.2f})"
    assert dark_recall > 0.80, f"word over the dark patch was dropped ({dark_recall:.2f})"


def test_stroke_cap_scales_with_text_size():
    """A fixed pixel cap silently rejects every glyph of the same text at a higher res."""
    from dsf.refine.strokes import stroke_bounds

    cfg = StrokeConfig()
    _, small = stroke_bounds(cfg, text_height=30)
    _, large = stroke_bounds(cfg, text_height=200)
    assert small == cfg.max_stroke, "small text keeps the configured floor"
    assert large > cfg.max_stroke * 4, "4K credits need a far wider cap"


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


def _faded_credit(opacity: float, width=900, height=420):
    """A credit composited at *opacity* over a lit, detailed scene."""
    from PIL import Image, ImageDraw

    from conftest import load_font

    rng = np.random.default_rng(5)
    coarse = cv2.resize(rng.normal(0, 1, (height // 24, width // 24)).astype(np.float32),
                        (width, height), interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.normal(0, 1, (height // 5, width // 5)).astype(np.float32),
                      (width, height), interpolation=cv2.INTER_CUBIC)
    bg = cv2.GaussianBlur(coarse, (0, 0), 9) + cv2.GaussianBlur(fine, (0, 0), 3) * 0.25
    bg = (bg - bg.min()) / (np.ptp(bg) + 1e-6)
    bg = np.clip((bg - 0.5) * 1.8 + 0.45, 0, 1)
    base = np.stack([bg * 235 + 10] * 3, -1).astype(np.uint8)

    font = load_font(40)
    text = "jonny lee miller"
    canvas = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bb = draw.textbbox((0, 0), text, font=font)
    x, y = (width - (bb[2] - bb[0])) // 2, int(height * 0.55)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(round(255 * opacity))))
    frame = np.array(Image.alpha_composite(canvas, layer).convert("RGB"))

    cover = Image.new("L", (width, height), 0)
    ImageDraw.Draw(cover).text((x, y), text, font=font, fill=255)
    box = (x - 8, y - 8, x + (bb[2] - bb[0]) + 8, y + (bb[3] - bb[1]) + 8)
    return frame, np.array(cover), box


def test_fading_credit_gives_a_whole_mask_not_a_shredded_one():
    """Credits that fade in used to come back in tatters.

    The mask was built by asking whether each pixel matched the colour of the other strokes,
    which only holds while the text is opaque. Mid-fade every pixel is part background, so
    one glyph reads as many shades and the test punched holes through it - worst exactly
    where the shot behind it varied most.
    """
    for opacity in (0.45, 0.6, 0.8):
        frame, cover, box = _faded_credit(opacity)
        patch = _extract(frame, box)
        assert patch is not None, f"nothing detected at opacity {opacity}"
        alpha = compose_alpha([patch], frame.shape[0], frame.shape[1])
        values = alpha[cover > 200]
        mean = float(values.mean())
        spread = float(values.std() / max(mean, 1e-6))
        assert spread < 0.35, \
            f"mask is uneven across the glyphs at opacity {opacity} (cv {spread:.2f})"
        assert float((values > 0.5 * mean).mean()) > 0.90, \
            f"holes through the text at opacity {opacity}"


def test_mask_strength_follows_the_fade():
    """A half-faded credit should push the depth half as far, so the text eases in."""
    levels = {}
    for opacity in (0.4, 0.7, 1.0):
        frame, cover, box = _faded_credit(opacity)
        patch = _extract(frame, box)
        assert patch is not None
        alpha = compose_alpha([patch], frame.shape[0], frame.shape[1])
        levels[opacity] = float(alpha[cover > 200].mean())

    for opacity, measured in levels.items():
        assert abs(measured - opacity) < 0.15, \
            f"opacity {opacity} came back as {measured:.2f}; masks: {levels}"
    assert levels[0.4] < levels[0.7] < levels[1.0]
