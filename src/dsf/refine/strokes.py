"""Turn a text *box* into a per-glyph *alpha mask*.

A detector box covers a whole text line, including the background between the letters.
Stamping that rectangle into the depth map would flatten a big slab of real scene depth, so
we go one level finer and pull out the strokes themselves.

The method is to estimate the picture *without* the writing - a median over a window wider
than a stroke - and read the text off as the difference. Dividing that difference by the
contrast between the text and the estimated background recovers the text's opacity directly,
which is what makes this survive both a box straddling a lighting boundary and a credit
part-way through a fade.

The mask is deliberately *soft* - glyph edges are anti-aliased in the source, a fading credit
is genuinely semi-transparent, and a hard binary edge stamped into a depth map produces
visible ringing after stereo warping.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from ..config import StrokeConfig
from ..detect.base import Detection


@dataclass
class AlphaPatch:
    """A soft alpha mask for one detection, stored as a crop to keep windows cheap."""

    x0: int
    y0: int
    alpha: np.ndarray  # float32 in [0, 1] - opacity, i.e. how strongly the text is showing
    det: Detection
    level: float = 1.0  # the opacity its strokes peak at; < 1 mid-fade

    @property
    def shape(self) -> tuple[int, int]:
        return self.alpha.shape[:2]

    @property
    def normalised(self) -> np.ndarray:
        """The stroke shape with the fade divided out, so strokes read as 1."""
        if self.level <= 1e-6:
            return self.alpha
        return np.clip(self.alpha / self.level, 0.0, 1.0)


_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold on an arbitrary float array, returned in the array's own units."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    peak = float(finite.max())
    if peak <= 1e-6:
        return 0.0
    u8 = np.clip(finite / peak * 255.0, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(thresh) / 255.0 * peak


def estimate_background(lum: np.ndarray, k: int) -> np.ndarray:
    """What the picture would look like with the writing taken off it.

    A median over a window a whole letter across: the writing is a minority of any such
    window, so it is voted away and what remains is the scene behind it. Sizing the window
    off a stroke rather than a letter is not enough - a bold stroke then fills more than half
    of it, wins its own median, and its interior is reported as background.

    A morphological opening is the textbook choice here and is wrong for this job. Opening
    takes the local *minimum*, so a glyph's dark outline drags the estimate down across a
    whole neighbourhood, and the ordinary picture around the text then reads as signal - a
    bright halo hugging every word. A median does not care which side a thin structure is
    on, so outline and fill are both simply outvoted.
    """
    k = max(3, int(k) | 1)
    u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
    return cv2.medianBlur(u8, k).astype(np.float32) / 255.0


def _residual_pair(lum: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(brighter-than-background, darker-than-background, the background itself)."""
    background = estimate_background(lum, k)
    difference = lum - background
    return (np.clip(difference, 0.0, None), np.clip(-difference, 0.0, None), background)


def _core(signal: np.ndarray) -> np.ndarray:
    """Everything the residual picked out, across the whole crop."""
    thresh = _otsu(signal)
    core = signal > thresh
    return core if int(core.sum()) >= 8 else np.zeros(signal.shape, bool)


def _centres(signal: np.ndarray) -> np.ndarray:
    """The strongest fifth of a residual - the middles of whatever it found."""
    core = _core(signal)
    if not core.any():
        return core
    return signal >= np.percentile(signal[core], 80)


def _colour_spread(lum: np.ndarray, signal: np.ndarray, core: np.ndarray) -> float:
    """How much the luma varies from one picked-out blob to the next.

    Burned-in text is one flat colour everywhere it appears. The gaps *between* its strokes
    are not text at all - they are whatever the picture is doing - so they inherit the shot's
    full range. That difference says which sign of the residual is the writing, and unlike
    response magnitude it does not care about contrast.

    Measured per blob and then compared across blobs. A single pooled sample cannot do it:
    pooling the peaks only samples wherever the background contrasts most, which is locally
    uniform and makes everything look flat, while pooling a whole core drags in each blob's
    soft edges, which sit at background luma and make everything look varied. Each blob's own
    centre is pure, and it is the variation *between* blobs that is diagnostic.
    """
    if not core.any():
        return 1.0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    medians = []
    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] < 8:
            continue
        blob = labels == label
        strength = signal[blob]
        values = lum[blob]
        medians.append(float(np.median(values[strength >= np.median(strength)])))
    if len(medians) < 2:
        pooled = lum[core]
        return float(np.percentile(pooled, 90) - np.percentile(pooled, 10))
    return float(np.percentile(medians, 90) - np.percentile(medians, 10))


def _enclosed_fraction(inner: np.ndarray, outer: np.ndarray) -> float:
    """How much of *inner* sits inside the region *outer* closes around.

    This is the asymmetry that tells a glyph from its own outline: an outline is a closed
    loop drawn around the fill, so filling its holes swallows the fill whole, while filling
    the fill's holes leaves the outline outside.

    Comparing what each one's immediate rim touches would be the obvious alternative, but a
    stroke's anti-aliased fringe pushes that rim into the background, which is exactly where
    the measurement breaks down. Hole-filling is a region test, so a one-pixel fringe cannot
    swing it.
    """
    if not inner.any() or not outer.any():
        return 0.0
    filled = binary_fill_holes(outer)
    if filled is None:
        return 0.0
    return float(filled[inner].mean())


@dataclass
class CropStats:
    """Background-free glyph response for one crop.

    Two masks, because the shape of the writing and how strongly it is showing are separate
    questions. Everything that reasons about *what* the strokes are - thresholding, stroke
    width, connected components - has to work on ``shape``, which is normalised so a stroke
    reads as 1 no matter how faint the text is. ``alpha`` is that shape scaled back down by
    ``level``, and it is what gets composited.
    """

    polarity: str
    alpha: np.ndarray  # float32 opacity in [0, 1] - the real strength of the text
    shape: np.ndarray  # float32 in [0, 1], normalised so strokes read as 1
    level: float  # the crop's peak opacity: 1.0 for solid text, lower mid-fade
    rim: np.ndarray  # thin structures of the opposite sign - the outline, if any


def _expand_into_rim(core: np.ndarray, rim: np.ndarray, iterations: int) -> np.ndarray:
    """Geodesic dilation of the glyph core into a hard outline drawn around it.

    Off by default, because it is only safe when the text really has a drawn outline. Title
    cards more often carry a soft drop shadow, and a shadow thresholds into a ragged region:
    growing into that crusts every glyph with speckle, which survives into the depth map as
    corroded-looking text. The heal step already repairs the depth around the strokes, so the
    outline's corruption is dealt with either way - following it only sharpens the result
    when the outline is genuinely hard-edged.
    """
    if iterations <= 0 or not rim.any() or not core.any():
        return core
    allowed = core | rim
    out = core
    for _ in range(iterations):
        grown = cv2.dilate(out.astype(np.uint8), _K3).astype(bool) & allowed
        if int(grown.sum()) == int(out.sum()):
            break
        out = grown
    return out


def _decide_polarity(lum: np.ndarray, white: np.ndarray, black: np.ndarray,
                     w_peak: float, b_peak: float, cfg: StrokeConfig) -> str:
    """Which sign of the residual is the writing.

    Both signs almost always respond. Text is thin, but so is every gap between its strokes,
    and outlined text answers one sign with its fill and the other with its rim - so response
    strength on its own decides nothing. Three questions, strongest first:

    1. Does only one sign respond at all? Rare, but free to check.
    2. Which one is contained by the other? Writing is figure and everything else is ground,
       whether "everything else" is an outline drawn around it or just the picture behind it.
       This is the question that actually distinguishes them.
    3. If containment is a wash, fall back to colour flatness: the strokes are one flat
       colour while the gaps between them inherit whatever the shot is doing.
    """
    ratio = w_peak / max(b_peak, 1e-6)
    if ratio > cfg.polarity_ratio:
        return "light"
    if ratio < 1.0 / cfg.polarity_ratio:
        return "dark"

    w_core, b_core = _core(white), _core(black)
    light_score = _enclosed_fraction(_centres(white), b_core)
    dark_score = _enclosed_fraction(_centres(black), w_core)
    # Containment is only evidence when something is actually contained. A drawn outline is
    # a closed loop and scores near 1; plain background is an open region that runs off the
    # edge of the crop, so it encloses nothing and both scores come back low. Low scores
    # therefore mean "not outlined text", not "the other one wins".
    if max(light_score, dark_score) >= cfg.enclosure_min and \
            abs(light_score - dark_score) > cfg.enclosure_margin:
        return "light" if light_score > dark_score else "dark"

    w_spread = _colour_spread(lum, white, w_core)
    b_spread = _colour_spread(lum, black, b_core)
    return "light" if w_spread <= b_spread else "dark"


def analyse_crop(lum: np.ndarray, cfg: StrokeConfig,
                 text_height: float | None = None) -> CropStats | None:
    """Turn a luma crop into a soft glyph alpha, independent of the background behind it.

    A global luma split cannot survive real footage: as soon as a detection box straddles a
    lighting boundary - a bright shoulder on one side, shadow on the other - the classes get
    spent describing the *background* rather than separating text from it, and the polarity
    decision inverts. Subtracting an estimate of the background first removes that whole
    class of failure.
    """
    height = float(text_height if text_height else lum.shape[0])
    k = max(3, int(round(height * cfg.background_scale)))
    k += 1 - (k % 2)  # median kernels must be odd
    white, black, background = _residual_pair(lum, k)

    w_peak = float(np.percentile(white, 99.5))
    b_peak = float(np.percentile(black, 99.5))
    if max(w_peak, b_peak) < cfg.min_response:
        return None  # nothing thin and contrasty in this crop - no text here

    polarity = cfg.polarity
    if polarity == "auto":
        polarity = _decide_polarity(lum, white, black, w_peak, b_peak, cfg)

    signal, rim_signal = (white, black) if polarity == "light" else (black, white)
    if float(signal.max()) < cfg.min_response:
        return None

    # Recover the text's opacity rather than measuring its colour.
    #
    # A pixel where text of colour T covers background B at opacity a reads back as
    # lum = a*T + (1-a)*B, so the residual against the background is exactly a*(T - B).
    # Dividing by (T - B) therefore returns a itself, free of whatever the picture does.
    #
    # Testing the colour instead - "is this pixel the same shade as the other strokes?" -
    # holds only while the text is fully opaque. The moment a credit fades, every pixel is
    # part background, so the same glyph reads as many different shades and the mask comes
    # back shredded, worst wherever the shot behind it varies most.
    #
    # It also gives the right answer for free: a credit at 30% opacity yields a 30% mask, so
    # the depth is pushed 30% of the way and the text eases in instead of snapping.
    reference = 1.0 if polarity == "light" else 0.0
    # Floored so that where the background is already as bright as the text - and the text
    # is therefore invisible - the division attenuates instead of amplifying noise.
    contrast = np.maximum(np.abs(reference - background), cfg.min_response)
    alpha = np.clip(signal / contrast, 0.0, 1.0).astype(np.float32)

    # Split "what shape is it" from "how strongly is it showing". Left as one map, a credit
    # at 40% opacity produces a mask that is 0.4 everywhere, and every downstream test that
    # asks `> 0.5` - the stroke-width filter, the component filter - throws the whole word
    # away. Normalising by the crop's own peak restores a shape those tests can read, and
    # the peak is carried alongside as the level to scale back down by.
    core = _core(alpha)
    if not core.any():
        return None
    level = float(np.percentile(alpha[core], 90))
    if level < cfg.min_response:
        return None
    level = min(level, 1.0)
    shape = np.clip(alpha / level, 0.0, 1.0).astype(np.float32)
    if not np.any(shape > 0.5):
        return None

    # Only treat the opposite sign as an outline when it is a real response. On text drawn
    # without a rim that channel is just noise, and growing into it would fur every glyph.
    rim = np.zeros(lum.shape, bool)
    if float(np.percentile(rim_signal, 99.5)) >= cfg.min_response:
        rim_core = _core(rim_signal)
        rim_centres = _centres(rim_signal)
        if rim_core.any() and rim_centres.any():
            # An outline is flat-coloured too, so hold it to the same test as the fill.
            # Without this the rim also captures the gaps between strokes, which respond
            # just as thinly, and the mask creeps out into the picture.
            rim_lum = float(np.median(lum[rim_centres]))
            rim = rim_core & (np.abs(lum - rim_lum) <= cfg.luma_tol)
    return CropStats(polarity=polarity, alpha=alpha, shape=shape, level=level,
                     rim=rim)


def _stroke_width(mask: np.ndarray) -> float:
    """Twice the peak of the distance transform ~ the stroke thickness in pixels."""
    if not mask.any():
        return 0.0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return float(dist.max()) * 2.0


def extract_patch(frame: np.ndarray, det: Detection,
                  cfg: StrokeConfig) -> AlphaPatch | None:
    """Extract a soft glyph alpha for one detection. Returns None if nothing survives."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = det.bbox
    x0, y0 = max(0, x0 - cfg.pad), max(0, y0 - cfg.pad)
    x1, y1 = min(w, x1 + cfg.pad), min(h, y1 + cfg.pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    crop = frame[y0:y1, x0:x1].astype(np.float32)
    lum = (0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]) / 255.0

    stats = analyse_crop(lum, cfg, text_height=det.height)
    if stats is None:
        return None

    # Identify the strokes on the normalised shape, so a faded credit is judged the same as
    # a solid one, then scale the survivors back to the opacity they are actually showing at.
    binary = (stats.shape > 0.5).astype(np.uint8)
    if not binary.any():
        return None

    kept = _filter_components(binary, cfg, det.height, shape=stats.shape)
    if kept is None:
        return None

    shape = stats.shape * kept
    if cfg.rim_expand > 0:
        grown = _expand_into_rim(shape > 0.5, stats.rim, cfg.rim_expand)
        shape = np.maximum(shape, grown.astype(np.float32))
    alpha = (shape * stats.level).astype(np.float32)
    if float(alpha.max()) <= 0.0:
        return None
    return AlphaPatch(x0=x0, y0=y0, alpha=alpha, det=det, level=stats.level)


def stroke_bounds(cfg: StrokeConfig, text_height: float) -> tuple[float, float]:
    """Plausible stroke thickness for text of this size, in pixels.

    Fixed pixel limits cannot work across resolutions: a cap that is generous for a 720p
    subtitle silently rejects every glyph of the same credit roll at 4K. Real typefaces put
    the stem somewhere under a third of the cap height, so scale with the detected text and
    keep the configured value as a floor for very small text.
    """
    if text_height <= 0:
        return cfg.min_stroke, cfg.max_stroke
    return cfg.min_stroke, max(cfg.max_stroke, cfg.max_stroke_frac * text_height)


def _filter_components(binary: np.ndarray, cfg: StrokeConfig, text_height: float = 0.0,
                       shape: np.ndarray | None = None) -> np.ndarray | None:
    """Drop connected components that do not look like glyph strokes.

    One of the tests is about strength rather than form: a credit fades as a whole, so every
    glyph in it shares one opacity, and ``shape`` is normalised so that opacity reads as 1.
    A blob markedly fainter than that is not part of the same text.

    This matters most part-way through a fade. The normalisation divides by whatever the
    text is showing at, so while it is faint the divisor is small and everything else in the
    frame is amplified with it - a lit building edge behind the credit crosses the threshold
    and lands in the mask as a speck.
    """
    ch, cw = binary.shape
    crop_area = ch * cw
    min_stroke, max_stroke = stroke_bounds(cfg, text_height)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    keep = np.zeros(binary.shape, dtype=np.float32)
    any_kept = False
    max_span_w, max_span_h = cw - 1, ch - 1

    for label in range(1, n):
        x, y, bw, bh, area = stats[label]
        if area < cfg.min_cc_area:
            continue
        if area > cfg.max_cc_area_frac * crop_area:
            continue  # a background slab, not a letter
        # A component that spans the whole crop is background bleed.
        if bw >= max_span_w and bh >= max_span_h:
            continue

        comp = (labels == label)
        if shape is not None and float(shape[comp].max()) < cfg.min_relative_strength:
            continue
        sw = _stroke_width(comp)
        if not (min_stroke <= sw <= max_stroke):
            continue

        keep[comp] = 1.0
        any_kept = True

    if not any_kept:
        return None

    # Let the soft edges of surviving strokes back in: dilate the keep-mask by one pixel so
    # anti-aliased boundary pixels (alpha between 0 and 0.5) are not clipped off.
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return keep


def compose_alpha(patches, height: int, width: int, normalised: bool = False) -> np.ndarray:
    """Merge per-detection patches into one full-frame float32 alpha.

    With *normalised* set, the fade is divided out first, giving the stroke shape on its own.
    """
    alpha = np.zeros((height, width), dtype=np.float32)
    for patch in patches:
        ph, pw = patch.shape
        y0, x0 = patch.y0, patch.x0
        y1, x1 = min(height, y0 + ph), min(width, x0 + pw)
        if y1 <= y0 or x1 <= x0:
            continue
        source = patch.normalised if normalised else patch.alpha
        view = alpha[y0:y1, x0:x1]
        np.maximum(view, source[: y1 - y0, : x1 - x0], out=view)
    return alpha


def compose_levels(patches, height: int, width: int) -> np.ndarray:
    """A per-pixel map of how strongly the text is showing, one region per detection.

    Collapsing the frame to a single number - the strongest patch, say - lets one bad
    reading speak for every word on screen. Keeping the level where it was measured means a
    misread box only distorts its own few hundred pixels.
    """
    levels = np.zeros((height, width), dtype=np.float32)
    for patch in patches:
        ph, pw = patch.shape
        y0, x0 = patch.y0, patch.x0
        y1, x1 = min(height, y0 + ph), min(width, x0 + pw)
        if y1 <= y0 or x1 <= x0:
            continue
        covered = patch.normalised[: y1 - y0, : x1 - x0] > 0.25
        view = levels[y0:y1, x0:x1]
        np.maximum(view, np.where(covered, np.float32(patch.level), np.float32(0.0)),
                   out=view)
    return levels
