"""Glyph extraction: the mask must hug the strokes, not the box."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

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


def test_mask_reaches_the_outline_when_rim_growth_is_enabled():
    """The dark rim is part of the overlay, so growing into it is right for outlined text.

    Opt-in rather than default: on the soft drop shadow that title cards usually carry, the
    same growth crusts every glyph with speckle.
    """
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 6, y0 - 6, x1 + 6, y1 + 6),
                     StrokeConfig(rim_expand=3))
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


def test_bold_unoutlined_credit_still_reads_as_light():
    """The trap both earlier attempts at the outlined-text bug fell into.

    Text drawn straight onto the picture answers one sign with its strokes and the other
    with the gaps between them, and the gaps are thin too - so any fix that leans harder on
    shape or containment starts reading the gaps as the writing and the credit comes back
    as a hollow negative of itself. Containment is genuinely silent here, and has to be:
    plain background is an open region that encloses nothing.
    """
    bg = gradient_background(900, 300)
    frame, fill, _ = draw_subtitle_full(bg, "HELLO WORLD", font_size=52, stroke=0,
                                        outline=(255, 255, 255))
    x0, y0, x1, y1 = text_bbox(fill)
    for margin in (6, 12, 20, 30):
        patch = _extract(frame, (x0 - margin, y0 - margin, x1 + margin, y1 + margin))
        assert patch is not None, f"nothing found at margin {margin}"
        alpha = compose_alpha([patch], 300, 900)
        gaps = (alpha > 0.5) & (fill == 0)
        assert float((alpha[fill > 200] > 0.5).mean()) > 0.85, \
            f"margin {margin}: the strokes themselves were not masked"
        assert int(gaps.sum()) < int((fill > 200).sum()), \
            f"margin {margin}: the mask followed the gaps between the strokes"


@pytest.mark.parametrize("seed,background,texture", [(3, 120, 80), (12, 120, 80), (13, 140, 70)])
def test_a_coloured_credit_over_a_busy_shot_is_not_read_inside_out(seed, background, texture):
    """Amber text on a mid-grey shot: the flatness test used to pick the gaps between strokes.

    Both signs respond here, and neither encloses the other - plain background is an open
    region - so the decision falls to which sign is the flatter colour. That was compared in
    raw luma, but the two signs answer at quite different strengths, and the same 0.05 spread
    means one thing on a stroke standing well clear of its background and something else on a
    sliver worth 0.05 in total. Measuring each sign's flatness against its own response
    instead was right 74 times out of 84 on a real graded credit, against 62 before.

    Judged on the stroke *shape* rather than the opacity it is stamped at: this credit is not
    white, and the opacity model measures light text against white, so a saturated colour
    comes back at a fraction of full strength however solid it really is. That is a known
    limitation of the model, and a separate question from whether the polarity decision
    picked out the writing or the holes in it.
    """
    rng = np.random.default_rng(seed)
    h, w = 300, 900
    noise = cv2.resize(rng.normal(0, 1, (h // 14, w // 14)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    noise = (noise - noise.min()) / (np.ptp(noise) + 1e-6)
    base = np.clip(np.stack([noise * texture + background] * 3, -1), 0, 255).astype(np.uint8)
    frame, fill, _ = draw_subtitle_full(base, "GRADED CREDIT", font_size=52, stroke=0,
                                        fill=(255, 190, 80), outline=(255, 190, 80))
    x0, y0, x1, y1 = text_bbox(fill)
    patch = _extract(frame, (x0 - 8, y0 - 8, x1 + 8, y1 + 8))
    assert patch is not None

    shape = compose_alpha([patch], h, w, normalised=True)
    on_text = float((shape[fill > 200] > 0.5).mean())
    off_text = int(((shape > 0.5) & (fill == 0)).sum())
    assert on_text > 0.80, f"the credit came back inverted ({on_text:.0%} on the glyphs)"
    assert off_text < int((fill > 200).sum()), "the mask followed the gaps, not the strokes"


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


def _shadowed_text(offset=(6, 6)):
    """White text with a soft drop shadow, the way a title card is usually built."""
    from PIL import Image, ImageDraw, ImageFilter

    from conftest import load_font

    w, h = 900, 260
    base = Image.fromarray(gradient_background(w, h))
    font = load_font(60)
    text = "executive producer"
    shadow = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(shadow)
    bb = draw.textbbox((0, 0), text, font=font)
    x, y = (w - (bb[2] - bb[0])) // 2, h // 3
    draw.text((x + offset[0], y + offset[1]), text, font=font, fill=255)
    shadow = shadow.filter(ImageFilter.GaussianBlur(4))

    frame = np.array(base).astype(np.float32)
    frame *= (1.0 - 0.75 * (np.array(shadow).astype(np.float32) / 255.0))[..., None]
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    ImageDraw.Draw(img).text((x, y), text, font=font, fill=(255, 255, 255))

    fill = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fill).text((x, y), text, font=font, fill=255)
    return np.array(img), np.array(fill)


def test_a_shadowed_glyph_comes_out_with_a_clean_edge():
    """The reported symptom: speckles crusted around every glyph.

    Growing into the rim is right for a hard drawn outline and wrong for the soft drop
    shadow a title card usually carries - a shadow thresholds into a ragged region, and
    following it leaves a crust that survives into the depth map as corroded text. Hence
    rim_expand defaults to off.
    """
    frame, fill = _shadowed_text()
    x0, y0, x1, y1 = text_bbox(fill)
    box = (x0 - 8, y0 - 8, x1 + 8, y1 + 8)

    clean = _extract(frame, box)                                  # default: no growth
    crusted = _extract(frame, box, StrokeConfig(rim_expand=3))    # follow the shadow
    assert clean is not None and crusted is not None

    def raggedness(alpha):
        binary = (alpha > 0.5).astype(np.uint8)
        perimeter = (cv2.morphologyEx(binary, cv2.MORPH_GRADIENT,
                                      np.ones((3, 3), np.uint8)) > 0).sum()
        return perimeter / max(np.sqrt(binary.sum()), 1)

    assert raggedness(clean.alpha) < raggedness(crusted.alpha),         "following a drop shadow should roughen the boundary, which is why it is off"

    alpha = compose_alpha([clean], frame.shape[0], frame.shape[1])
    assert float((alpha[fill > 200] > 0.5).mean()) > 0.85,         "the glyphs themselves must still be covered"


def test_growing_into_an_outline_is_available_when_asked_for():
    """Still the right thing for genuinely outlined text - it just has to be opted into."""
    bg = gradient_background(900, 300)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=52, stroke=4)
    x0, y0, x1, y1 = text_bbox(fill)
    box = (x0 - 8, y0 - 8, x1 + 8, y1 + 8)

    off = compose_alpha([_extract(frame, box)], 300, 900)
    on = compose_alpha([_extract(frame, box, StrokeConfig(rim_expand=3))], 300, 900)
    rim = (full > 0) & (fill == 0)
    assert float((on[rim] > 0.5).mean()) > float((off[rim] > 0.5).mean()) + 0.15,         "rim_expand should measurably pull the drawn outline into the mask"


def _fading_text_over_detail(opacity):
    """A credit part-way through a fade, over a scene with bright detail beside it."""
    from PIL import Image, ImageDraw

    from conftest import load_font

    w, h = 900, 300
    rng = np.random.default_rng(11)
    scene = cv2.resize(rng.normal(0, 1, (h // 20, w // 20)).astype(np.float32), (w, h),
                       interpolation=cv2.INTER_CUBIC)
    scene = (scene - scene.min()) / (np.ptp(scene) + 1e-6)
    base = np.stack([scene * 90 + 40] * 3, -1).astype(np.uint8)
    # A lit edge next to the text - the kind of thing a faint fade amplifies into speckle.
    cv2.rectangle(base, (760, 120), (774, 210), (215, 215, 215), -1)
    base = cv2.GaussianBlur(base, (0, 0), 1.2)

    font = load_font(56)
    text = "executive producer"
    canvas = Image.fromarray(base).convert("RGBA")
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    bb = draw.textbbox((0, 0), text, font=font)
    x, y = 60, 130
    draw.text((x, y), text, font=font, fill=(255, 255, 255, int(round(255 * opacity))))
    frame = np.array(Image.alpha_composite(canvas, layer).convert("RGB"))

    cover = Image.new("L", (w, h), 0)
    ImageDraw.Draw(cover).text((x, y), text, font=font, fill=255)
    box = (x - 10, y - 10, x + (bb[2] - bb[0]) + 200, y + (bb[3] - bb[1]) + 10)
    return frame, np.array(cover), box


def test_a_fade_does_not_drag_scene_detail_into_the_mask():
    """The reported symptom: specks appearing *outside* the glyphs during a fade-in.

    The mask is normalised by whatever the text is showing at, so while it is faint the
    divisor is small and everything else in the frame is amplified with it - a lit edge
    behind the credit crosses the threshold and lands in the mask as a speck.
    """
    frame, cover, box = _fading_text_over_detail(0.45)
    gated = _extract(frame, box)
    ungated = _extract(frame, box, StrokeConfig(min_relative_strength=0.0))
    assert gated is not None and ungated is not None

    h, w = frame.shape[:2]
    text_area = cv2.dilate((cover > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)

    def off_text(patch):
        alpha = compose_alpha([patch], h, w)
        level = max(float(alpha.max()), 1e-6)
        return int(((alpha > 0.35 * level) & ~text_area).sum())

    assert off_text(gated) < off_text(ungated), \
        "the strength gate should remove scene detail the fade amplified"
    # ...without costing the text itself.
    alpha = compose_alpha([gated], h, w)
    assert float((alpha[cover > 200] > 0.2).mean()) > 0.85


def test_the_strength_gate_keeps_solid_text_intact():
    """Every glyph of a credit shares one opacity, so none of them should be near the gate."""
    bg = gradient_background(900, 300)
    frame, fill, _ = draw_subtitle_full(bg, "HELLO WORLD", font_size=52, stroke=0,
                                        outline=(255, 255, 255))
    x0, y0, x1, y1 = text_bbox(fill)
    box = (x0 - 8, y0 - 8, x1 + 8, y1 + 8)
    strict = compose_alpha([_extract(frame, box, StrokeConfig(min_relative_strength=0.9))],
                           300, 900)
    assert float((strict[fill > 200] > 0.5).mean()) > 0.85


def _bevelled_wordmark(w=520, h=150):
    """A pale card carrying letters with a dark edge and a lighter core.

    Real logotypes are shaded like this, and the opacity model reads one text colour, so
    the two tones come back as two different opacities within a single solid letter.
    """
    frame = np.full((h, w, 3), 210, dtype=np.uint8)
    frame[..., 2] = 180                                     # pale, slightly warm card
    for x in range(40, w - 40, 70):
        cv2.rectangle(frame, (x, 40), (x + 44, h - 40), (20, 40, 110), -1)      # dark edge
        cv2.rectangle(frame, (x + 10, 50), (x + 34, h - 50), (70, 110, 170), -1)  # lighter core
    return frame, (30, 30, w - 30, h - 30)


def _body_of(patch):
    """The inside of the glyphs, away from their antialiased edges."""
    shape = patch.normalised
    from scipy.ndimage import binary_fill_holes
    solid = binary_fill_holes(shape > 0.5)
    return cv2.erode(solid.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool), shape


def test_solidify_paints_a_two_tone_glyph_at_one_strength():
    """Otherwise the lighter half of every letter is stamped as though half transparent.

    The corrupted depth then shows through in patches across the wordmark, which is the
    artefact this exists to remove.
    """
    frame, box = _bevelled_wordmark()
    on = _extract(frame, box, StrokeConfig(polarity="dark"))
    off = _extract(frame, box, StrokeConfig(polarity="dark", solidify=False))
    assert on is not None and off is not None

    body_off, shape_off = _body_of(off)
    body_on, shape_on = _body_of(on)
    assert body_off.any() and body_on.any()
    uneven_off = float((shape_off[body_off] < 0.8).mean())
    uneven_on = float((shape_on[body_on] < 0.8).mean())
    assert uneven_off > 0.15, f"the fixture is not two-toned enough ({uneven_off:.0%})"
    assert uneven_on < uneven_off / 2, \
        f"solidify left {uneven_on:.0%} of the body faint, against {uneven_off:.0%} without it"


def test_solidify_does_not_paint_the_hole_in_a_letter():
    """The gap inside a D or an O is background and must keep its own depth."""
    w, h = 320, 180
    frame = np.full((h, w, 3), 210, dtype=np.uint8)
    frame[..., 2] = 180
    cv2.rectangle(frame, (60, 40), (260, 140), (20, 40, 110), -1)   # a ring...
    cv2.rectangle(frame, (85, 60), (235, 120), (210, 210, 180), -1)  # ...around a counter
    patch = _extract(frame, (30, 20, 290, 160), StrokeConfig(polarity="dark"))
    assert patch is not None
    # patch coordinates: the crop starts pad px outside the box
    counter = patch.alpha[60:100, 85:175]
    assert float(counter.max()) < 0.3, \
        f"the counter was painted at {counter.max():.2f}"


def test_a_background_window_larger_than_the_crop_does_not_blow_up():
    """OpenCV's constant-time median refuses very large kernels, and where it gives up
    depends on the pixel values rather than only on the size."""
    from dsf.refine.strokes import estimate_background

    rng = np.random.default_rng(0)
    for shape in ((101, 472), (40, 90), (7, 900)):
        lum = (rng.random(shape) * 0.2 + 0.7).astype(np.float32)
        for k in (3, 51, 401, 4001):
            out = estimate_background(lum, k)
            assert out.shape == lum.shape


def test_outlined_text_resolves_to_its_fill_however_the_box_is_cropped():
    """White-on-black-outline is the commonest subtitle style, and it used to invert.

    An outline sits between the fill and the background, so it answers the opposite sign
    louder than the writing does - the ratio measured 0.622 against a 0.625 cut-off. Which
    side of that it landed on depended on how tightly the detector had cropped the line, so
    the same subtitle came back as its letters at one box size and as the hollow ring around
    them at the next, and the depth plane was stamped onto the ring.

    Fixed by making the magnitude question mean what it always claimed to: "did the other
    sign respond at all", rather than "did it respond 1.6x less". It was being asked first,
    so it was overruling a containment test that scored this 1.00 to 0.18 at every margin
    here. Two earlier attempts reordered the questions instead and traded this for dropping
    bold unoutlined credits - `test_bold_unoutlined_credit_still_reads_as_light` is those
    attempts' failure, kept as a test so a third one cannot repeat it.
    """
    bg = gradient_background(640, 360)
    frame, fill, full = draw_subtitle_full(bg, "HELLO WORLD", font_size=36)
    x0, y0, x1, y1 = text_bbox(fill)
    fill_mask = fill > 200
    outline = (full > 0) & ~fill_mask

    for margin in (4, 8, 12, 16, 24, 32):
        for scale in (0.9, 1.5):
            patch = _extract(frame, (x0 - margin, y0 - margin, x1 + margin, y1 + margin),
                             StrokeConfig(background_scale=scale))
            assert patch is not None, f"nothing found at margin {margin}, scale {scale}"
            alpha = compose_alpha([patch], frame.shape[0], frame.shape[1])
            on_fill = float(alpha[fill_mask].mean())
            on_outline = float(alpha[outline].mean())
            assert on_fill > on_outline, (
                f"margin {margin}, background_scale {scale}: the mask followed the outline "
                f"({on_outline:.2f}) instead of the fill ({on_fill:.2f})")
