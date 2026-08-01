"""Regression tests against a real DepthCrafter pair, if one is present.

A synthetic credit can be made to look however the author expects. These run on actual
frames - a title card fading in and out over a moving crowd - and are skipped when the
footage is not on this machine.

Point DSF_REAL_FOOTAGE at a folder holding ``rgb_png/`` and ``depth_png/``.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(os.environ.get("DSF_REAL_FOOTAGE", r"F:\_3D_Test_\depth_sub_fixer"))
RGB, DEPTH = ROOT / "rgb_png", ROOT / "depth_png"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not (RGB.is_dir() and DEPTH.is_dir()),
                       reason=f"no real footage at {ROOT}"),
]

# The credit occupies this band; frame 0 of the sequence is before it appears and the last
# frame is after it has gone.
BAND = (slice(575, 740), slice(580, 1360))
SOLID = 20  # an index where the credit is fully opaque


def _config():
    from dsf.config import PipelineConfig, apply_profile

    cfg = apply_profile(PipelineConfig(), "both")
    return dataclasses.replace(cfg, filters=dataclasses.replace(cfg.filters, roi="full"))


@pytest.fixture(scope="module")
def masks():
    from dsf.config import configure_model_cache

    configure_model_cache()
    from dsf.media import probe
    from dsf.pipeline import masks_for_frames
    from dsf.temporal import from_u8

    info = probe(RGB)
    wanted = [0, 1, 2, 3, 4, 8, SOLID, 40, 64, 66, 68, 69, info.nb_frames - 1]
    got = masks_for_frames(str(RGB), _config(), wanted, info)
    return {i: from_u8(m) for i, (m, _) in got.items()}, info


def _glyph_reference(masks):
    """The credit is static, so a fully-opaque frame gives the shape to score against."""
    return masks[SOLID][BAND] > 0.5


def test_the_credit_is_found_when_it_is_solid(masks):
    got, _ = masks
    reference = _glyph_reference(got)
    assert reference.sum() > 5000, "the credit should cover thousands of pixels"
    assert float(np.median(got[SOLID][BAND][reference])) > 0.85


def test_frames_with_no_credit_are_left_completely_alone(masks):
    """Depth on either side of the fade was never corrupted; touching it is a regression."""
    got, info = masks
    for index in (0, info.nb_frames - 1):
        assert float(got[index].max()) == 0.0, \
            f"frame index {index} has no credit but produced a mask"


def test_the_bold_word_is_masked_as_solidly_as_the_rest(masks):
    """`jonny` is bold. Sizing the background window off a stroke instead of a letter let
    the bold word win its own median, and it came back hollow - or vanished entirely."""
    got, _ = masks
    reference = _glyph_reference(got)
    columns = np.arange(reference.shape[1])[None, :]
    xs = np.nonzero(reference.any(axis=0))[0]
    split = xs.min() + int(0.32 * (xs.max() - xs.min()))
    bold = reference & (columns < split)
    rest = reference & (columns >= split)
    assert bold.sum() > 1000 and rest.sum() > 1000

    for index in (4, 8, 66, 68):
        band = got[index][BAND]
        bold_level = float(np.median(band[bold]))
        rest_level = float(np.median(band[rest]))
        assert bold_level > 0.4 * rest_level, (
            f"frame index {index}: bold word masked at {bold_level:.2f} against "
            f"{rest_level:.2f} for the rest of the line"
        )


def test_mask_strength_rises_and_falls_with_the_fade(masks):
    """Mid-fade the mask should be part strength, and it should move in one direction."""
    got, _ = masks
    reference = _glyph_reference(got)
    level = {i: float(np.median(got[i][BAND][reference])) for i in (0, 1, 2, 3, 4, SOLID)}
    rising = [level[i] for i in (0, 1, 2, 3, 4, SOLID)]
    assert rising == sorted(rising), f"fade-in should not go backwards: {rising}"
    assert rising[0] == 0.0 and rising[-1] > 0.85
    assert 0.0 < level[2] < 0.9, "a part-faded credit should give a part-strength mask"


def test_masks_track_the_credits_real_opacity(masks):
    """Measured against the actual pixels, not against an assumption about the fade."""
    got, _ = masks
    reference = _glyph_reference(got)

    def true_opacity(index: int) -> float:
        files = sorted(RGB.glob("*.png"))
        crop = cv2.imread(str(files[index]))[BAND].astype(np.float32)
        lum = (0.114 * crop[..., 0] + 0.587 * crop[..., 1] + 0.299 * crop[..., 2]) / 255.0
        background = cv2.medianBlur((lum * 255).astype(np.uint8), 41).astype(np.float32) / 255.0
        return max(0.0, float(np.median((lum - background)[reference])))

    for index in (1, 2, 3, 4):
        measured = float(np.median(got[index][BAND][reference]))
        actual = true_opacity(index)
        assert abs(measured - actual) < 0.25, (
            f"frame index {index}: mask {measured:.2f} vs measured opacity {actual:.2f}"
        )


def test_end_to_end_render_preserves_every_untouched_pixel(tmp_path):
    """A sequence in must give a sequence out, same names, same dtype, edits only on text."""
    from dsf.config import configure_model_cache

    configure_model_cache()
    from dsf.pipeline import run_fix

    out = tmp_path / "fixed"
    result = run_fix(str(RGB), str(DEPTH), str(out), _config(), max_frames=6)
    assert result["frames"] == 6

    src = sorted(DEPTH.glob("*.png"))[:6]
    dst = sorted(out.glob("*.png"))
    assert [p.name for p in dst] == [p.name for p in src]

    before = cv2.imread(str(src[0]), cv2.IMREAD_UNCHANGED)
    after = cv2.imread(str(dst[0]), cv2.IMREAD_UNCHANGED)
    assert before.dtype == after.dtype and before.shape == after.shape
    # Index 0 is ahead of the credit, so it must come back untouched.
    np.testing.assert_array_equal(before, after)
