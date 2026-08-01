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


def heal_edt(depth: np.ndarray, mask: np.ndarray, smooth: float = 2.0) -> np.ndarray:
    """Fill *mask* from the nearest valid depth pixel, then soften the seam.

    Deliberately not cv2.inpaint: that is an 8-bit-only API and would quantise a 10-bit
    depth map down to 256 levels. A Euclidean-distance nearest-valid fill keeps full
    precision and costs about the same.
    """
    if not mask.any() or mask.all():
        return depth
    indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
    filled = depth[tuple(indices)]
    if smooth > 0:
        blurred = cv2.GaussianBlur(filled, (0, 0), float(smooth))
        filled = np.where(mask, blurred, filled)
    return np.where(mask, filled, depth).astype(np.float32)


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
            heal_area = cv2.dilate(text.astype(np.uint8), _ellipse(cfg.heal_dilate)).astype(bool)
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
