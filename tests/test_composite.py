"""Depth repair maths."""

from __future__ import annotations

import numpy as np
import pytest

from dsf.composite import (
    ALPHA_EPS, brightness_to_code, code_range, composite_frame, heal_edt, region_mask,
    resize_alpha, resolve_range,
)
from dsf.config import CompositeConfig


@pytest.mark.parametrize("bit_depth,vrange,expected", [
    (8, "tv", (16.0, 235.0)),
    (10, "tv", (64.0, 940.0)),
    (12, "tv", (256.0, 3760.0)),
    (8, "pc", (0.0, 255.0)),
    (10, "pc", (0.0, 1023.0)),
])
def test_code_range(bit_depth, vrange, expected):
    assert code_range(bit_depth, vrange) == expected


def test_brightness_maps_into_the_legal_range():
    assert brightness_to_code(0.0, 10, "tv") == 64.0
    assert brightness_to_code(1.0, 10, "tv") == 940.0
    assert brightness_to_code(0.5, 10, "tv") == pytest.approx(502.0)
    # Out-of-range input is clamped rather than producing an illegal code.
    assert brightness_to_code(2.0, 10, "tv") == 940.0
    assert brightness_to_code(-1.0, 10, "pc") == 0.0


def test_resolve_range_prefers_explicit_override():
    assert resolve_range(CompositeConfig(value_range="auto"), "pc") == "pc"
    assert resolve_range(CompositeConfig(value_range="tv"), "pc") == "tv"


def test_empty_mask_returns_the_input_untouched():
    depth = np.full((32, 32), 500, dtype=np.uint16)
    alpha = np.zeros((32, 32), dtype=np.float32)
    out = composite_frame(depth, alpha, CompositeConfig(), 10, "tv")
    np.testing.assert_array_equal(out, depth)


def test_full_alpha_paints_the_requested_code():
    depth = np.full((32, 32), 500, dtype=np.uint16)
    alpha = np.ones((32, 32), dtype=np.float32)
    cfg = CompositeConfig(brightness=1.0, heal="none", dilate=0, feather=0.0)
    out = composite_frame(depth, alpha, cfg, 10, "tv")
    assert int(out.min()) == 940 and int(out.max()) == 940


def test_pixels_outside_the_mask_are_not_modified():
    rng = np.random.default_rng(1)
    depth = rng.integers(64, 940, (64, 64)).astype(np.uint16)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[20:24, 20:40] = 1.0
    cfg = CompositeConfig(brightness=0.9, heal="none", dilate=0, feather=0.0)
    out = composite_frame(depth, alpha, cfg, 10, "tv")

    untouched = alpha <= ALPHA_EPS
    np.testing.assert_array_equal(out[untouched], depth[untouched])


def test_alpha_blends_proportionally():
    depth = np.full((8, 8), 100, dtype=np.uint16)
    alpha = np.full((8, 8), 0.5, dtype=np.float32)
    cfg = CompositeConfig(brightness=1.0, heal="none", dilate=0, feather=0.0, value_range="pc")
    out = composite_frame(depth, alpha, cfg, 10, "pc")
    assert int(out[0, 0]) == pytest.approx(round(100 * 0.5 + 1023 * 0.5), abs=1)


def test_output_never_exceeds_the_bit_depth():
    depth = np.full((16, 16), 1023, dtype=np.uint16)
    alpha = np.ones((16, 16), dtype=np.float32)
    cfg = CompositeConfig(brightness=1.0, heal="none", value_range="pc")
    out = composite_frame(depth, alpha, cfg, 10, "pc")
    assert int(out.max()) <= 1023


def test_heal_replaces_a_corrupted_patch_with_its_surroundings():
    depth = np.full((64, 64), 400.0, dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 28:36] = True
    depth[mask] = 1000.0  # the DepthCrafter artefact

    healed = heal_edt(depth, mask, smooth=1.0)
    assert healed[mask].max() < 500.0, "artefact should be replaced by nearby depth"
    np.testing.assert_allclose(healed[~mask], depth[~mask])


def test_heal_removes_the_smear_around_the_glyphs():
    """End to end: a bright halo around text must be gone after compositing."""
    depth = np.full((80, 80), 300, dtype=np.uint16)
    alpha = np.zeros((80, 80), dtype=np.float32)
    alpha[38:42, 30:50] = 1.0          # the glyph
    depth[34:46, 26:54] = 900          # the smear DepthCrafter bled around it

    cfg = CompositeConfig(brightness=0.5, heal="edt", heal_scope="glyph", heal_dilate=8,
                          dilate=0, feather=0.0)
    out = composite_frame(depth, alpha, cfg, 10, "tv")

    halo = np.zeros((80, 80), dtype=bool)
    halo[34:46, 26:54] = True
    halo[38:42, 30:50] = False  # exclude the painted glyph itself
    assert int(out[halo].max()) < 500, "smear should have been healed away"
    assert int(out[39, 40]) == pytest.approx(502, abs=2), "glyph painted at 0.5 brightness"


def test_region_scope_fills_whole_boxes():
    mask = np.zeros((32, 32), dtype=bool)
    mask[10, 10] = True
    mask[14, 20] = True
    filled = region_mask(mask)
    assert filled[10, 10] and filled[14, 20]
    assert filled.sum() == 2, "separate components get separate boxes"

    joined = np.zeros((32, 32), dtype=bool)
    joined[10:12, 10:20] = True
    assert region_mask(joined).sum() == 20


def test_relative_brightness_tracks_the_local_depth():
    depth = np.full((64, 64), 200, dtype=np.uint16)
    alpha = np.zeros((64, 64), dtype=np.float32)
    alpha[30:34, 20:44] = 1.0
    cfg = CompositeConfig(brightness_mode="relative", relative_offset=0.1, heal="none",
                          dilate=0, feather=0.0)
    out = composite_frame(depth, alpha, cfg, 10, "tv")
    painted = int(out[32, 30])
    assert painted > 200, "text should sit in front of the surrounding depth"
    assert painted == pytest.approx(200 + 0.1 * (940 - 64), abs=2)


def test_resize_alpha_matches_depth_resolution():
    alpha = np.zeros((360, 640), dtype=np.float32)
    alpha[100:120, 200:400] = 1.0
    out = resize_alpha(alpha, 320, 180)
    assert out.shape == (180, 320)
    assert out.max() == pytest.approx(1.0, abs=0.01)
    assert resize_alpha(alpha, 640, 360) is alpha
