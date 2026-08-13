"""Depth repair: heal the corrupted halo, then paint the glyphs.

Two separate problems get fixed here. DepthCrafter does not just get the glyph pixels wrong -
it bleeds a smeared halo into the depth around them, so simply overwriting the letters would
leave a visible ring of bad depth. Step one heals that halo from neighbouring valid depth.
Step two stamps the glyph shape at the requested grey level, so the text sits on one clean,
constant depth plane.

All arithmetic happens in float32 and is written back at the source bit depth. Pixels outside
the mask are returned untouched.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import CompositeConfig

#: alpha below this is treated as "not text" when deciding what to heal
ALPHA_EPS = 0.02


def code_range(bit_depth: int, value_range: str) -> tuple[float, float]:
    """Legal luma code range for the given bit depth and colour range."""
    if value_range == "pc":
        return 0.0, float((1 << bit_depth) - 1)
    if value_range == "tv":
        shift = bit_depth - 8
        return float(16 << shift), float(235 << shift)
    raise ValueError(f"value_range must be 'tv' or 'pc'; got {value_range!r}")


def resolve_range(cfg: CompositeConfig, source_range: str) -> str:
    return source_range if cfg.value_range == "auto" else cfg.value_range


def brightness_to_code(brightness: float, bit_depth: int, value_range: str) -> float:
    lo, hi = code_range(bit_depth, value_range)
    return lo + float(np.clip(brightness, 0.0, 1.0)) * (hi - lo)


def _ellipse(radius: int) -> np.ndarray:
    k = max(1, 2 * int(radius) + 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _blur_radius(sigma: float) -> int:
    """Half-width of the kernel cv2.GaussianBlur picks for itself at this sigma."""
    return (int(round(float(sigma) * 8 + 1)) | 1) // 2


def _window(mask: np.ndarray, margin: int) -> tuple[slice, slice]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (slice(max(0, int(ys.min()) - margin), min(h, int(ys.max()) + 1 + margin)),
            slice(max(0, int(xs.min()) - margin), min(w, int(xs.max()) + 1 + margin)))


def heal_edt(depth: np.ndarray, mask: np.ndarray, smooth: float = 2.0) -> np.ndarray:
    """Fill *mask* from the nearest valid depth pixel, then soften the seam.

    Deliberately not cv2.inpaint: that is an 8-bit-only API and would quantise a 10-bit
    depth map down to 256 levels. A Euclidean-distance nearest-valid fill keeps full
    precision and costs about the same.

    The transform runs on a window around the text rather than the whole frame. Subtitles
    and credits occupy a band - on the test clip the mask's bounding box is 0.05 of the
    frame's 1.54 megapixels - and a distance transform costs its input, so the full frame
    was thirty times the work the answer needed.

    The window is grown until every filled pixel found its source closer than the window's
    own edge, which is what makes the crop exact rather than merely close: a source outside
    the window would have to be further away than one already inside it.
    """
    if not mask.any() or mask.all():
        return depth
    # Wide enough that the blur below reads real neighbours rather than the crop's edge.
    margin = max(32, 2 * _blur_radius(smooth))
    while True:
        box = _window(mask, margin)
        sub_mask = mask[box]
        distances, indices = distance_transform_edt(sub_mask, return_indices=True)
        if float(distances.max()) < margin or sub_mask.shape == mask.shape:
            break
        margin *= 2

    sub_depth = depth[box]
    filled = sub_depth[tuple(indices)]
    if smooth > 0:
        blurred = cv2.GaussianBlur(filled, (0, 0), float(smooth))
        filled = np.where(sub_mask, blurred, filled)
    out = depth.astype(np.float32, copy=True)
    out[box] = np.where(sub_mask, filled, sub_depth)
    return out


def heal_radius(text: np.ndarray, cfg: CompositeConfig) -> int:
    """How far past the strokes to repair, measured off the strokes themselves.

    DepthCrafter's smear is not a fixed number of pixels; it scales with the thing being
    smeared. So a constant radius is wrong twice over - it means one thing on a depth map
    that came back at half the clip's resolution and another at full, and one thing on a
    subtitle and another on a title card. Measured on a credit whose mask strokes ran 8.2 px
    of a 960x384 map, the corruption was still 10 codes out at thirteen pixels while the
    configured 6 cleared nothing beyond two.

    Twice the peak of the distance transform is the stroke thickness, the same measure the
    extractor filters components on - written out here rather than imported, because the
    compositor otherwise knows nothing about glyph extraction and this is two lines of it.
    The configured value stays as a floor, so setting it high still wins on a clip that wants
    more than the strokes imply.
    """
    if not text.any():
        return max(0, int(cfg.heal_dilate))
    stroke = 2.0 * float(cv2.distanceTransform(text.astype(np.uint8), cv2.DIST_L2, 3).max())
    return max(int(cfg.heal_dilate), int(round(float(cfg.heal_strokes) * stroke)))


def region_mask(mask: np.ndarray) -> np.ndarray:
    """Bounding-box fill of each connected component - the aggressive heal scope."""
    out = np.zeros_like(mask, dtype=bool)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for label in range(1, n):
        x, y, w, h, _ = stats[label]
        out[y:y + h, x:x + w] = True
    return out


def shape_alpha(alpha: np.ndarray, cfg: CompositeConfig) -> np.ndarray:
    """Grow and feather the glyph alpha before it is stamped."""
    a = alpha
    if cfg.dilate > 0:
        a = cv2.dilate(a, _ellipse(cfg.dilate))
    if cfg.feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(cfg.feather))
    return np.clip(a, 0.0, 1.0)


def _relative_code(depth: np.ndarray, mask: np.ndarray, cfg: CompositeConfig,
                   lo: float, hi: float) -> float:
    """Place the text just in front of the depth immediately surrounding it."""
    ring = cv2.dilate(mask.astype(np.uint8), _ellipse(12)).astype(bool) & \
        ~cv2.dilate(mask.astype(np.uint8), _ellipse(4)).astype(bool)
    sample = depth[ring] if ring.any() else depth[~mask]
    if sample.size == 0:
        return lo + cfg.brightness * (hi - lo)
    base = float(np.percentile(sample, 90))
    return float(np.clip(base + cfg.relative_offset * (hi - lo), lo, hi))


def composite_frame(depth_y: np.ndarray, alpha: np.ndarray, cfg: CompositeConfig,
                    bit_depth: int, source_range: str) -> np.ndarray:
    """Return a repaired copy of one depth luma plane.

    Args:
        depth_y: uint16 HxW luma codes straight from the decoder.
        alpha: float32 HxW glyph mask in [0, 1], already at depth-map resolution.
        cfg: compositing options.
        bit_depth: source bit depth (8, 10, 12...).
        source_range: "tv" or "pc", from the probe.
    """
    if depth_y.shape != alpha.shape:
        raise ValueError(f"depth {depth_y.shape} and alpha {alpha.shape} shapes differ")

    max_code = (1 << bit_depth) - 1
    if not np.any(alpha > ALPHA_EPS):
        return depth_y.astype(np.uint16, copy=False)

    vrange = resolve_range(cfg, source_range)
    lo, hi = code_range(bit_depth, vrange)
    depth = depth_y.astype(np.float32)
    text = alpha > ALPHA_EPS

    if cfg.heal != "none":
        if cfg.heal_scope == "region":
            heal_area = region_mask(text)
        elif cfg.heal_scope == "glyph":
            heal_area = cv2.dilate(text.astype(np.uint8),
                                   _ellipse(heal_radius(text, cfg))).astype(bool)
        else:
            raise ValueError(f"unknown heal_scope {cfg.heal_scope!r}; use glyph or region")
        if cfg.heal == "edt":
            depth = heal_edt(depth, heal_area, cfg.heal_smooth)
        else:
            raise ValueError(f"unknown heal mode {cfg.heal!r}; use edt or none")

    if cfg.brightness_mode == "relative":
        value = _relative_code(depth, text, cfg, lo, hi)
    elif cfg.brightness_mode == "absolute":
        value = lo + float(np.clip(cfg.brightness, 0.0, 1.0)) * (hi - lo)
    else:
        raise ValueError(f"unknown brightness_mode {cfg.brightness_mode!r}")

    a = shape_alpha(alpha, cfg)
    out = depth * (1.0 - a) + value * a
    return np.clip(np.rint(out), 0, max_code).astype(np.uint16)


def resize_alpha(alpha: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resample a soft alpha mask to the depth map's resolution.

    DepthCrafter routinely outputs at a different resolution than the source clip. INTER_AREA
    when downscaling preserves the anti-aliased stroke edges instead of aliasing them away.
    """
    if alpha.shape[:2] == (height, width):
        return alpha
    src_h, src_w = alpha.shape[:2]
    interp = cv2.INTER_AREA if (width < src_w or height < src_h) else cv2.INTER_LINEAR
    return cv2.resize(alpha, (width, height), interpolation=interp).astype(np.float32)
