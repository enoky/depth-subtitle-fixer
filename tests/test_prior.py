"""The temporal prior: marks must show up as real text somewhere nearby in time."""

from __future__ import annotations

import dataclasses

import numpy as np

from dsf.config import PipelineConfig, apply_profile
from dsf.pipeline import context_radius, remembered

H, W = 120, 320


def _frame(boxes, level):
    """One (shape, level, detections) triple. *boxes* are (y, x, h, w) rectangles."""
    shape = np.zeros((H, W), np.uint8)
    levels = np.zeros((H, W), np.uint8)
    for y, x, h, w in boxes:
        shape[y:y + h, x:x + w] = 255
        levels[y:y + h, x:x + w] = int(round(level * 255))
    return shape, levels, []


def _run(frames, cfg):
    return list(remembered(iter(frames), cfg))


def _cfg(**temporal):
    base = PipelineConfig()
    return dataclasses.replace(base, temporal=dataclasses.replace(base.temporal, **temporal))


GLYPH = (40, 60, 20, 90)      # where the credit sits
STRAY = (20, 240, 8, 10)      # a lit edge elsewhere in the shot


def test_a_mark_seen_only_while_the_text_is_faint_is_dropped():
    """The reported symptom: specks that appear only during the fade-in."""
    frames = []
    for i in range(21):
        level = min(1.0, 0.15 + 0.09 * i)
        boxes = [GLYPH] + ([STRAY] if level < 0.4 else [])
        frames.append(_frame(boxes, level))

    out = _run(frames, _cfg())
    y, x, h, w = STRAY
    gy, gx, gh, gw = GLYPH
    faint = out[0][0]
    assert faint[y:y + h, x:x + w].max() == 0, "the speck should not survive"
    assert faint[gy:gy + gh, gx:gx + gw].min() == 255, "the credit itself must be untouched"


def test_the_text_itself_is_never_touched():
    frames = [_frame([GLYPH], min(1.0, 0.15 + 0.09 * i)) for i in range(21)]
    out = _run(frames, _cfg())
    gy, gx, gh, gw = GLYPH
    for shape, _, _ in out:
        assert shape[gy:gy + gh, gx:gx + gw].min() == 255


def test_scrolling_credits_are_left_alone():
    """The hazard the overlap check exists for.

    A remembered shape sits at fixed pixels. Text that moves would land outside it, and
    filtering against it would erase the credit rather than clean it up - so the prior only
    filters a frame it already explains.
    """
    frames = []
    for i in range(21):
        level = min(1.0, 0.15 + 0.09 * i)
        frames.append(_frame([(8 + 4 * i, 60, 20, 90)], level))

    out = _run(frames, _cfg())
    for i, (shape, _, _) in enumerate(out):
        y = 8 + 4 * i
        assert shape[y:y + 20, 60:150].min() == 255, \
            f"frame {i}: scrolling text was erased by the prior"


def test_the_soft_edge_of_a_glyph_survives_a_dropped_speck():
    """Masks are soft, and the antialiased skirt sits below the threshold blobs are cut at.

    Judging blobs and then keeping only blob pixels quietly shaves that skirt off every
    glyph in the frame - a few percent of the mask, on every frame the prior acts on, with
    nothing in the blob arithmetic to show for it.
    """
    gy, gx, gh, gw = GLYPH
    frames = []
    for i in range(21):
        level = min(1.0, 0.15 + 0.09 * i)
        shape, levels, _ = _frame([GLYPH] + ([STRAY] if level < 0.4 else []), level)
        # A two-step ramp out of the glyph, both steps under the blob threshold.
        shape[gy - 2:gy + gh + 2, gx - 2:gx + gw + 2] = np.maximum(
            shape[gy - 2:gy + gh + 2, gx - 2:gx + gw + 2], 40)
        shape[gy - 1:gy + gh + 1, gx - 1:gx + gw + 1] = np.maximum(
            shape[gy - 1:gy + gh + 1, gx - 1:gx + gw + 1], 100)
        frames.append((shape, levels, []))

    faint = _run(frames, _cfg())[0][0]
    y, x, h, w = STRAY
    assert faint[y:y + h, x:x + w].max() == 0, "the speck should still be dropped"
    assert faint[gy - 1, gx + 10] == 100, "the glyph's soft edge was shaved off"
    assert faint[gy - 2, gx + 10] == 40, "the glyph's soft edge was shaved off"


def test_nothing_is_filtered_without_confident_evidence():
    """A clip that never gets above the evidence level has no grounds to overrule itself."""
    frames = [_frame([GLYPH, STRAY], 0.25) for _ in range(21)]
    out = _run(frames, _cfg())
    y, x, h, w = STRAY
    assert out[0][0][y:y + h, x:x + w].max() == 255


def test_the_prior_can_be_switched_off():
    frames = []
    for i in range(21):
        level = min(1.0, 0.15 + 0.09 * i)
        frames.append(_frame([GLYPH] + ([STRAY] if level < 0.4 else []), level))
    out = _run(frames, _cfg(prior_window=0))
    y, x, h, w = STRAY
    assert out[0][0][y:y + h, x:x + w].max() == 255, "off means off"
    assert len(out) == len(frames)


def test_switching_the_prior_off_shrinks_the_preview_window():
    """It is the reason a preview costs what it does, so turning it off must buy that back."""
    on = apply_profile(PipelineConfig(), "both")
    off = dataclasses.replace(on, temporal=dataclasses.replace(on.temporal, prior_window=0))
    assert context_radius(on) > context_radius(off)


def test_every_frame_survives_the_stage():
    frames = [_frame([GLYPH], 0.9) for _ in range(7)]
    assert len(_run(frames, _cfg())) == 7
