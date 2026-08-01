"""Turn a text *box* into a per-glyph *alpha mask*.

A detector box covers a whole text line, including the background between the letters.
Stamping that rectangle into the depth map would flatten a big slab of real scene depth, so
we go one level finer and pull out the strokes themselves.

Burned-in subtitles are built to be legible: bright flat glyphs, a dark outline or drop
shadow, hard edges. That makes them separable inside the box by luminance alone, which is
both faster and more precise on thin strokes than asking a general segmenter.

The mask is deliberately *soft* - glyph edges are anti-aliased in the source, and a hard
binary edge stamped into a depth map produces visible ringing after stereo warping.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import StrokeConfig
from ..detect.base import Detection


@dataclass
class AlphaPatch:
    """A soft alpha mask for one detection, stored as a crop to keep windows cheap."""

    x0: int
    y0: int
    alpha: np.ndarray  # float32 in [0, 1]
    det: Detection

    @property
    def shape(self) -> tuple[int, int]:
        return self.alpha.shape[:2]


_K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def _otsu_split(lum: np.ndarray) -> tuple[float, float, float]:
    """Otsu threshold on a [0,1] luma crop -> (threshold, dark mean, bright mean)."""
    u8 = np.clip(lum * 255.0, 0, 255).astype(np.uint8)
    thresh, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(thresh) / 255.0
    dark, bright = lum[lum <= t], lum[lum > t]
    dark_mean = float(dark.mean()) if dark.size else 0.0
    bright_mean = float(bright.mean()) if bright.size else 1.0
    return t, dark_mean, bright_mean


def _three_way_split(lum: np.ndarray) -> tuple[float, float]:
    """Two thresholds separating outline / background / glyph.

    A plain two-class Otsu is the wrong tool here. A subtitle crop has *three* populations -
    the dark outline, the video behind it, and the bright glyph fill - and Otsu habitually
    puts the cut between the outline and everything else, which makes the glyphs look like
    part of the majority class and flips the polarity decision.
    """
    from skimage.filters import threshold_multiotsu

    try:
        t1, t2 = threshold_multiotsu(lum, classes=3, nbins=256)
        return float(t1), float(t2)
    except (ValueError, RuntimeError):
        t, _, _ = _otsu_split(lum)  # degenerate crop (flat, or only two levels)
        return t, t


def _enclosure(core: np.ndarray, other: np.ndarray, background: np.ndarray) -> float:
    """How completely *other* wraps *core*, penalised by background leaking into the rim.

    This is the one asymmetry that separates a glyph from its own outline. The fill is
    sealed inside the outline, so its rim is outline and nothing else. The outline is a
    loop: its inner rim is fill but its outer rim is the video behind it, so background
    shows up and drags the score down.

    Comparing raw luma distances instead would be worse than useless - a black outline is
    always further from a mid-grey background than white text is, so the outline would win
    every time.
    """
    if not core.any():
        return -2.0
    ring = cv2.dilate(core.astype(np.uint8), _K3).astype(bool) & ~core
    if int(ring.sum()) < 4:
        return -2.0
    return float(other[ring].mean()) - float(background[ring].mean())


@dataclass
class CropStats:
    """Where the glyphs sit in the luma histogram of one crop."""

    polarity: str
    lo: float  # luma mapped to alpha 0
    hi: float  # luma mapped to alpha 1
    rim: np.ndarray  # the opposite extreme class - the outline, if there is one
    rim_is_majority: bool  # if so it is background, not an outline, and must not be grown into


def _expand_into_rim(core: np.ndarray, rim: np.ndarray, iterations: int) -> np.ndarray:
    """Geodesic dilation of the glyph core into its own outline.

    The luma ramp deliberately cuts off before the background, which also clips the dark
    rim the glyphs are drawn with. That rim is part of the burned-in text and DepthCrafter
    gets its depth just as wrong, so grow into it - but only into rim pixels, and only a
    few steps, so a dark *background* can never be flooded.
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


def analyse_crop(lum: np.ndarray, cfg: StrokeConfig) -> CropStats | None:
    """Decide polarity and the two luma anchors for the soft alpha ramp."""
    t1, t2 = _three_way_split(lum)
    hi_mask, lo_mask = lum > t2, lum < t1
    if not hi_mask.any() and not lo_mask.any():
        return None
    mid_mask = ~hi_mask & ~lo_mask

    # The background is whichever class owns the most pixels. Taking the middle class on
    # faith fails whenever the outline and the background land in the same bin - then the
    # middle holds nothing but anti-aliased edges, and its mean is a value that appears
    # nowhere in the image.
    named = {"lo": lo_mask, "mid": mid_mask, "hi": hi_mask}
    majority_key = max(named, key=lambda k: int(named[k].sum()))
    bg_mask = named[majority_key]
    background = float(lum[bg_mask].mean())
    hi_mean = float(lum[hi_mask].mean()) if hi_mask.any() else background
    lo_mean = float(lum[lo_mask].mean()) if lo_mask.any() else background

    polarity = cfg.polarity
    if polarity == "auto":
        # Text is never the bulk of a text box, so the majority class cannot be the glyphs.
        # When outline and background merge into one class this alone settles it.
        candidates = [p for p in ("light", "dark")
                      if majority_key != ("hi" if p == "light" else "lo")]
        if len(candidates) == 1:
            polarity = candidates[0]
        else:
            light = _enclosure(hi_mask, lo_mask, bg_mask)
            dark = _enclosure(lo_mask, hi_mask, bg_mask)
            if abs(light - dark) > 0.1:
                polarity = "light" if light > dark else "dark"
            else:
                # Nothing is enclosing anything - text drawn without an outline. Fall back
                # to whichever extreme sits further from the background.
                polarity = "light" if (hi_mean - background) >= (background - lo_mean) \
                    else "dark"

    if polarity == "light":
        lo, hi = 0.5 * (background + t2), hi_mean
        rim, rim_key = lo_mask, "lo"
    else:
        lo, hi = 0.5 * (background + t1), lo_mean
        rim, rim_key = hi_mask, "hi"
    if abs(hi - lo) < 1e-3:
        return None
    return CropStats(polarity=polarity, lo=lo, hi=hi, rim=rim,
                     rim_is_majority=(majority_key == rim_key))


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

    stats = analyse_crop(lum, cfg)
    if stats is None:
        return None
    polarity = stats.polarity
    alpha = np.clip((lum - stats.lo) / (stats.hi - stats.lo), 0.0, 1.0).astype(np.float32)

    binary = (alpha > 0.5).astype(np.uint8)
    if not binary.any():
        return None

    kept = _filter_components(binary, lum, alpha, cfg, polarity)
    if kept is None:
        return None

    alpha = alpha * kept
    if cfg.rim_expand > 0 and not stats.rim_is_majority:
        grown = _expand_into_rim(alpha > 0.5, stats.rim, cfg.rim_expand)
        alpha = np.maximum(alpha, grown.astype(np.float32))
    if float(alpha.max()) <= 0.0:
        return None
    return AlphaPatch(x0=x0, y0=y0, alpha=alpha, det=det)


def _filter_components(binary: np.ndarray, lum: np.ndarray, alpha: np.ndarray,
                       cfg: StrokeConfig, polarity: str) -> np.ndarray | None:
    """Drop connected components that do not look like glyph strokes."""
    ch, cw = binary.shape
    crop_area = ch * cw
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return None

    keep = np.zeros(binary.shape, dtype=np.float32)
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
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
        sw = _stroke_width(comp)
        if not (cfg.min_stroke <= sw <= cfg.max_stroke):
            continue

        if cfg.outline_check and not _has_outline(comp, lum, ring_kernel, cfg, polarity):
            continue

        keep[comp] = 1.0
        any_kept = True

    if not any_kept:
        return None

    # Let the soft edges of surviving strokes back in: dilate the keep-mask by one pixel so
    # anti-aliased boundary pixels (alpha between 0 and 0.5) are not clipped off.
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return keep


def _has_outline(comp: np.ndarray, lum: np.ndarray, kernel: np.ndarray,
                 cfg: StrokeConfig, polarity: str) -> bool:
    """Burned-in text carries a dark outline or drop shadow. Bright scenery does not."""
    core = comp.astype(np.uint8)
    ring = cv2.dilate(core, kernel).astype(bool) & ~comp
    if ring.sum() < 4:
        return False
    core_mean = float(lum[comp].mean())
    ring_mean = float(lum[ring].mean())
    if polarity == "light":
        return (core_mean - ring_mean) > cfg.outline_delta
    return (ring_mean - core_mean) > cfg.outline_delta


def compose_alpha(patches, height: int, width: int) -> np.ndarray:
    """Merge per-detection patches into one full-frame float32 alpha."""
    alpha = np.zeros((height, width), dtype=np.float32)
    for patch in patches:
        ph, pw = patch.shape
        y0, x0 = patch.y0, patch.x0
        y1, x1 = min(height, y0 + ph), min(width, x0 + pw)
        if y1 <= y0 or x1 <= x0:
            continue
        view = alpha[y0:y1, x0:x1]
        np.maximum(view, patch.alpha[: y1 - y0, : x1 - x0], out=view)
    return alpha
