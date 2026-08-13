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
from scipy.ndimage import binary_erosion, binary_fill_holes

from ..config import StrokeConfig
from ..detect.base import Detection
from ..filters import chromaticity


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
    # Past twice the crop's shorter side the window is entirely border replication in that
    # axis and tells you nothing the edge rows do not. It is also where OpenCV's
    # constant-time median starts refusing outright - on a 101x472 crop it raised `k < 16`
    # at 393, and where exactly it gives up depends on the values, not just the size. So
    # this is a ceiling on what the crop can answer rather than a guess at that limit.
    k = min(k, max(3, (2 * min(u8.shape) - 1) | 1))
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


def _solidify(shape: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Paint a glyph's body at full strength, leaving its antialiased edge soft.

    The opacity model reads one text colour. A bevelled logotype has two - a dark edge and a
    lighter core - so the core comes back as though the glyph were half transparent there,
    and the corrupted depth underneath shows through in patches across every letter. The
    glyph is not half transparent; it is two colours, and its body is entirely covered by
    it. Measured on a real title card, a third of the body sat below 0.8 before this and an
    eighth after.

    Only pixels strictly inside the body are raised, so the boundary keeps the soft value
    that stops the stamp ringing. Counters - the enclosed gaps in letters like D and O - are
    background, lie outside the body, and are never touched.
    """
    body = binary_erosion(shape > threshold, np.ones((3, 3), bool), border_value=0)
    return np.maximum(shape, body.astype(np.float32))


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

    Question 1 asks whether the other sign responded, and nothing more. It used to ask
    whether this one responded 1.6 times harder, which is a different and much weaker
    question - and it was decided *first*, so it overruled a containment test that was
    answering correctly and decisively. White-on-black-outline, the commonest subtitle style
    there is, produces a ratio of about 0.62 against that 0.625 cut-off: which side of it a
    given line landed on turned on how tightly the detector had cropped the box, so the same
    subtitle came back as its letters at one margin and as the hollow ring around them at the
    next. Containment separated those two readings 1.00 to 0.18 every single time.
    """
    # Not a ratio: `analyse_crop` has already established that at least one sign clears
    # `min_response`, so this is "the other one is nothing but noise", and on noise the two
    # questions below have nothing to work with.
    if b_peak < cfg.min_response:
        return "light"
    if w_peak < cfg.min_response:
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

    # Each sign's flatness measured against its own response, not in raw luma. The two signs
    # routinely answer at quite different strengths, and a 0.05 luma spread means one thing
    # on a stroke standing 0.7 clear of its background and something else entirely on a
    # sliver worth 0.05 in total - so comparing the raw numbers asks a different question of
    # each side. On an amber credit whose letters carry a vertical gradient, the raw form
    # read the writing as *less* flat than the fragments between it and inverted: over 84
    # boxes of that credit it was right 62 times against 74 for this.
    w_spread = _colour_spread(lum, white, w_core) / max(w_peak, cfg.min_response)
    b_spread = _colour_spread(lum, black, b_core) / max(b_peak, cfg.min_response)
    return "light" if w_spread <= b_spread else "dark"


#: A luma level this close to solid has nothing left to recover, so the colour pass is not
#: run at all. White text - which is most text - lands here, so the common case pays nothing.
_OPAQUE_ENOUGH = 0.95

#: And below this the colour pass is not run either, because it may refine a reading but it
#: must not rescue one. The same 0.5 the rest of the pipeline uses to call a pixel text at
#: all, so what it will adjust is exactly what is already being masked.
#:
#: The demo clip is why. It carries a cyan shop sign the camera photographed, which is meant
#: to keep its real depth, and the appearance gate lets it through - it was only ever
#: harmless because its luma reading of 0.38 put it under the threshold at which a mask does
#: anything. Reading it in colour lifts it to 0.62 and it lands in the mask as a slab: the
#: correction is not wrong about the sign, which really is an opaque colour, but a filmed
#: sign scoring higher is worth nothing here and being masked is the one failure this tool
#: advertises avoiding. That the gate lets it through at all is a separate bug this uncovered.
#:
#: The cost is a step, not a slope: a *coloured* credit part-way through a fade crosses this
#: line and its mask jumps. Fades are quick and the temporal filter spans them, and the
#: alternative was every solid coloured credit stamped at three quarters for ever.
_TOO_FAINT_TO_REREAD = 0.5


def _opacity_in_colour(rgb: np.ndarray | None, k: int, core: np.ndarray, polarity: str,
                       cfg: StrokeConfig) -> float | None:
    """How opaque the text is, asked of each colour channel and answered by the loudest.

    The opacity model divides the residual by the contrast between the text and the
    background it covers, and it takes the text's colour as pure white for light text or
    pure black for dark. That is exactly right for a subtitle and wrong for everything else:
    an amber credit at luma 0.77 over a shot at 0.15 divides by 0.85 where it should divide
    by 0.62, so a fully opaque credit reports 0.76 and a quarter of the corruption it was
    meant to bury shows back through it. Measured on a real credit, which read 0.76 against
    a predicted 0.73.

    Asking each channel separately fixes it without giving up what the luma model was
    protecting. A bright saturated colour is bright because some channel is at or near its
    maximum - amber is (255, 190, 80), so in red it *is* white - and that channel's reading
    is the honest one. Taking the loudest is therefore reading the opacity off whichever
    channel the white reference happens to be true for.

    Crucially it does nothing to a fade. White text answers identically in all three
    channels, so the maximum is the luma answer; a white credit at 45% reports 45% in red,
    green and blue alike. And a *coloured* credit mid-fade is handled correctly too: amber at
    half strength puts its red channel half way from the background to full, which is 0.5.
    What it cannot do is separate opaque mid-grey text from half-strength white, because
    nothing can - they are the same pixels.
    """
    if rgb is None:
        return None
    light = polarity == "light"
    reference = 1.0 if light else 0.0
    best = 0.0
    for channel in range(rgb.shape[2]):
        plane = rgb[..., channel]
        base = estimate_background(plane, k)
        residual = plane - base if light else base - plane
        contrast = np.maximum(np.abs(reference - base), cfg.min_response)
        answer = np.clip(residual / contrast, 0.0, 1.0)[core]
        if answer.size:
            best = max(best, float(np.percentile(answer, 90)))
    return min(best, 1.0)


def analyse_crop(lum: np.ndarray, cfg: StrokeConfig,
                 text_height: float | None = None,
                 rgb: np.ndarray | None = None) -> CropStats | None:
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

    # The shape is normalised by the luma reading above, which is the right divisor for it;
    # what it gets scaled back by is asked again in colour, because luma alone systematically
    # under-reads any text that is not white. Skipped once the luma answer is already solid,
    # which is where white text lands, so the common case pays nothing for this.
    if _TOO_FAINT_TO_REREAD <= level < _OPAQUE_ENOUGH:
        strength = _opacity_in_colour(rgb, k, core, polarity, cfg)
        if strength is not None:
            level = max(level, strength)

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


#: Blobs a crop needs before the agreement tests will speak at all. With two there is no
#: majority to be the odd one out of.
_MIN_CLUSTER = 3

#: How many times the blobs' own scatter a blob must be out by, on top of the configured
#: floor, before it is rejected. It exists for the shots where the slab is genuinely uneven,
#: and it has to stay well under the floor on ordinary ones or it silently becomes the only
#: bar there is and `depth_tol` stops doing anything.
#:
#: It was 8, which did exactly that. The number came from measuring how far a blob sits from
#: its line's median - but that measurement was taken while the polarity decision was
#: inverting on the clip it was taken from, so the "blobs" being measured were the gaps
#: between the glyphs rather than the glyphs. Gaps are scattered across whatever the picture
#: is doing and needed a wide bar; real letters sit on the slab together and do not. With
#: that fixed the typical line scatters 0.017, so 8x put the bar at 0.136 - above the 0.10
#: floor, past double the worst deviation a genuine letter manages (0.089), and unmoved by
#: the tolerance the user is offered. At 3x it lands near 0.05 and the floor governs again.
_SCATTER_K = 3.0

#: Share of the crop's median blob area below which a blob neither votes nor can be rejected.
#: A depth map arrives blurred and often at a lower resolution than the picture, so a mark
#: much smaller than the letters around it has no depth of its own to read - what comes back
#: is the halo over it. The full stop ending a subtitle is the case that showed this: at a
#: twentieth the area of the letters it measured a depth of its own that no letter shared,
#: and was thrown out of every line that ended in one. Small *intruders* escaping with it is
#: the accepted cost, and they are what `min_cc_area` and the temporal prior are already for.
#:
#: The half is measured, against clips with the glyphs labelled. Removing the guard entirely
#: costs 399 px of letters on a credit and 1125 px on a clip with nothing in it to reject at
#: all - every one of those a false positive by construction. A quarter still costs 481. At
#: a half nothing text is touched on either, and raising it further only gives up background:
#: 2076 px caught at 0.8 against 2725 here. So this is the smallest value that leaves the
#: writing alone, which is the side to err on. The test that covers it pins the guard's
#: existence rather than its exact value - a synthetic mark is either resolvable or it is
#: not, while where to put the line between the two is a question only real footage answers.
_MIN_JUDGED_AREA_FRAC = 0.5


def _blob_median(values: np.ndarray, blob: np.ndarray, strength: np.ndarray,
                 valid: np.ndarray | None = None, min_samples: int = 8):
    """Median of *values* over the strong half of one blob, or None if it cannot answer.

    The strong half rather than the whole blob, for the same reason `_colour_spread` samples
    that way: a stroke's outer pixels are part background by construction, so including them
    drags every blob's reading toward the picture behind it and makes all of them look alike -
    which is precisely the difference these tests are trying to measure.
    """
    sel = blob if valid is None else (blob & valid)
    if int(sel.sum()) < min_samples:
        return None
    weight = strength[sel]
    return np.median(values[sel][weight >= np.median(weight)], axis=0)


def _keep_agreeing(blobs: list[np.ndarray], values: np.ndarray | None, tol: float,
                   min_agree: float, strength: np.ndarray,
                   valid: np.ndarray | None = None) -> list[np.ndarray]:
    """Drop the blobs whose median *values* disagree with the value most blobs share.

    One line of burned-in text is one thing: one flat colour, and - once DepthCrafter has
    pasted it onto a slab of wrong depth - one depth. Its glyphs therefore agree with each
    other on both. An object *behind* the text that happens to match its luma is under no
    such obligation, and luma is all the residual can see. This is the argument
    `_colour_spread` makes about the sign of the residual, asked one blob at a time and on
    the axes where the luma test has nothing left to say.

    A veto, never a requirement, and it gives ground twice over.

    The bar it holds blobs to is the configured floor *or* several times the blobs' own
    scatter, whichever is larger. A fixed distance cannot do this job: how flat a slab
    DepthCrafter lays over a given credit varies from shot to shot, so the same tolerance
    that is generous on one line is measuring quantisation noise on the next. Scaling by
    what the line's own letters do makes the question "is this blob out by more than the
    others manage between themselves", which is the question actually being asked.

    And it stands down entirely whenever the majority does not agree even at that bar,
    because that means the crop is not reading one flat thing: text too small for
    DepthCrafter to have responded to takes the depth of whatever is behind it, and a wall
    receding across the shot then hands every letter a different answer. Rejecting on that
    would bite the far end off the line, so a scattered vote means "this crop cannot answer",
    not "keep whichever blob sat at the median".
    """
    if values is None or tol <= 0 or len(blobs) < _MIN_CLUSTER:
        return blobs
    areas = np.array([int(blob.sum()) for blob in blobs], dtype=np.float32)
    judged = areas >= _MIN_JUDGED_AREA_FRAC * float(np.median(areas))
    medians = [_blob_median(values, blob, strength, valid) if ok else None
               for blob, ok in zip(blobs, judged)]
    answered = [i for i, m in enumerate(medians) if m is not None]
    if len(answered) < _MIN_CLUSTER:
        return blobs

    stack = np.asarray([medians[i] for i in answered], dtype=np.float32)
    # Median rather than mean, and no assumption about which way the odd one out lies: the
    # convention for whether near is bright or dark is the depth model's business, not ours.
    delta = stack - np.median(stack, axis=0)
    distance = np.abs(delta) if delta.ndim == 1 else np.linalg.norm(delta, axis=1)
    bar = max(float(tol), _SCATTER_K * float(np.median(distance)))
    agrees = distance <= bar
    if float(agrees.mean()) < min_agree:
        return blobs

    rejected = {answered[j] for j, ok in enumerate(agrees) if not ok}
    return [blob for i, blob in enumerate(blobs) if i not in rejected]


def extract_patch(frame: np.ndarray, det: Detection, cfg: StrokeConfig,
                  depth: np.ndarray | None = None) -> AlphaPatch | None:
    """Extract a soft glyph alpha for one detection. Returns None if nothing survives.

    *depth* is the corrupted depth map itself, as a full-frame float32 in [0, 1] at this
    frame's resolution - `dsf.pipeline.depth_guide` builds one. It is only ever used to
    reject blobs that cannot be part of the same text, never to find text: the map is the
    thing being repaired, and how strongly it responded to any given credit is not something
    to bet a glyph on. Without it the extraction is exactly what it was.
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = det.bbox
    x0, y0 = max(0, x0 - cfg.pad), max(0, y0 - cfg.pad)
    x1, y1 = min(w, x1 + cfg.pad), min(h, y1 + cfg.pad)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None

    crop = frame[y0:y1, x0:x1].astype(np.float32)
    lum = (0.2126 * crop[..., 0] + 0.7152 * crop[..., 1] + 0.0722 * crop[..., 2]) / 255.0

    stats = analyse_crop(lum, cfg, text_height=det.height, rgb=crop / 255.0)
    if stats is None:
        return None

    # Identify the strokes on the normalised shape, so a faded credit is judged the same as
    # a solid one, then scale the survivors back to the opacity they are actually showing at.
    binary = (stats.shape > 0.5).astype(np.uint8)
    if not binary.any():
        return None

    kept = _filter_components(binary, cfg, det.height, shape=stats.shape,
                              depth=depth[y0:y1, x0:x1] if depth is not None else None,
                              chroma=chromaticity(crop) if cfg.chroma_tol > 0 else None)
    if kept is None:
        return None

    shape = stats.shape * kept
    # After the component filter, never before it: that filter asks how strong each blob is
    # relative to the strongest text in the box, and filling a blob to 1 first would hand
    # every speck the fade amplified the same answer as the text.
    if cfg.solidify:
        shape = _solidify(shape)
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
                       shape: np.ndarray | None = None,
                       depth: np.ndarray | None = None,
                       chroma: tuple[np.ndarray, np.ndarray] | None = None
                       ) -> np.ndarray | None:
    """Drop connected components that do not look like glyph strokes.

    Two kinds of test, in two passes. The first asks of each blob on its own whether it is
    shaped and lit like a stroke - area, span, thickness, and strength relative to the
    strongest text in the same crop, since a credit fades as a whole and ``shape`` is
    normalised so that shared opacity reads as 1. That last one matters most part-way through
    a fade: the normalisation divides by whatever the text is showing at, so while it is
    faint the divisor is small and everything else in the frame is amplified with it - a lit
    building edge behind the credit crosses the threshold and lands in the mask as a speck.

    The second pass asks whether the survivors agree with *each other*, on the axes the luma
    residual is blind to. A background object the same brightness as the text answers the
    residual exactly as a glyph does and passes every test in the first pass; what gives it
    away is that it is not at the text's depth, and often not its colour either. See
    `_keep_agreeing`. Both are optional and both stand down rather than guess.
    """
    ch, cw = binary.shape
    crop_area = ch * cw
    min_stroke, max_stroke = stroke_bounds(cfg, text_height)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    max_span_w, max_span_h = cw - 1, ch - 1
    blobs: list[np.ndarray] = []

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

        blobs.append(comp)

    if not blobs:
        return None

    # Where a blob's reading is sampled from. Without a normalised shape every pixel of a
    # blob counts equally, which is what the tests that call this directly expect.
    strength = shape if shape is not None else binary.astype(np.float32)
    blobs = _keep_agreeing(blobs, depth, cfg.depth_tol, cfg.cluster_min_agree, strength)
    if chroma is not None:
        blobs = _keep_agreeing(blobs, chroma[0], cfg.chroma_tol, cfg.cluster_min_agree,
                               strength, valid=chroma[1])
    if not blobs:
        return None

    keep = np.zeros(binary.shape, dtype=np.float32)
    for blob in blobs:
        keep[blob] = 1.0

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
