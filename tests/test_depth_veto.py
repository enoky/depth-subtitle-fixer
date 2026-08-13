"""The mask must hug the writing, not everything in the box that is the same brightness.

The residual is measured on luma alone, so an object behind the text that happens to match
its brightness answers exactly as a glyph does: right area, right thickness, right strength.
Nothing in the luma channel separates them. Two things do - burned-in text sits on one flat
slab of wrong depth, and it is one flat colour - and both are asked the same way: does this
blob agree with the blobs around it?
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from conftest import load_font
from dsf.config import PipelineConfig, StrokeConfig
from dsf.detect.base import Detection, bbox_to_poly
from dsf.refine.strokes import compose_alpha, extract_patch

W, H = 900, 300


def _scene_with_lookalikes(colour=(255, 255, 255)):
    """A white credit, plus objects in the same box that answer the residual just as loudly.

    Returns ``(frame, glyph_cover, object_cover, box)``. The objects are bars rather than
    noise on purpose: a speck is thrown out by the area and strength gates long before any
    of this, and the failure being reproduced here is the one where the intruder passes
    every test a single blob can be given.
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(3)
    scene = cv2.resize(rng.normal(0, 1, (H // 20, W // 20)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_CUBIC)
    scene = (scene - scene.min()) / (np.ptp(scene) + 1e-6)
    base = np.stack([scene * 70 + 30] * 3, -1).astype(np.uint8)

    objects = np.zeros((H, W), np.uint8)
    for x in (735, 775, 815):
        cv2.rectangle(objects, (x, 120), (x + 11, 205), 255, -1)
    base[objects > 0] = colour
    base = cv2.GaussianBlur(base, (0, 0), 0.8)

    font = load_font(52)
    text = "executive producer"
    img = Image.fromarray(base)
    draw = ImageDraw.Draw(img)
    bb = draw.textbbox((0, 0), text, font=font)
    x, y = 60, 120
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    cover = Image.new("L", (W, H), 0)
    ImageDraw.Draw(cover).text((x, y), text, font=font, fill=255)
    box = (x - 10, y - 10, 840, y + (bb[3] - bb[1]) + 10)
    return np.array(img), np.array(cover), objects, box


def _slab_guide(cover: np.ndarray, text_depth=0.80, scene_depth=0.30,
                scale: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """Depth the way DepthCrafter returns it over burned-in text: a flat slab, plus halo.

    *scale* renders the slab at a fraction of the picture's resolution before stretching it
    back, which is the normal case rather than an awkward one - depth maps routinely come
    back smaller than the clip they describe, and it is what stops a small mark holding a
    depth of its own.
    """
    guide = np.full((H, W), scene_depth, np.float32)
    halo = cv2.dilate((cover > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    guide[halo] = text_depth
    if scale != 1.0:
        small = cv2.resize(guide, (int(W * scale), int(H * scale)),
                           interpolation=cv2.INTER_AREA)
        guide = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(guide, (0, 0), sigma)


def _extract(frame, box, cfg=None, depth=None):
    det = Detection(poly=bbox_to_poly(*box), score=1.0)
    return extract_patch(frame, det, cfg or StrokeConfig(), depth=depth)


def _covered(patch, mask: np.ndarray) -> float:
    alpha = compose_alpha([patch], H, W)
    return float((alpha[mask] > 0.5).mean())


# --------------------------------------------------------------- the depth agreement veto

def test_an_object_the_colour_of_the_text_is_dropped_once_depth_is_known():
    """The reported failure, and the fix. Luma cannot tell these apart; depth can."""
    frame, glyphs, objects, box = _scene_with_lookalikes()
    blind = _extract(frame, box)
    seeing = _extract(frame, box, depth=_slab_guide(glyphs))
    assert blind is not None and seeing is not None

    on_objects_blind = _covered(blind, objects > 0)
    on_objects_seeing = _covered(seeing, objects > 0)
    assert on_objects_blind > 0.5, \
        f"the fixture is not reproducing the bug ({on_objects_blind:.0%} of the objects masked)"
    assert on_objects_seeing < 0.05, \
        f"depth left {on_objects_seeing:.0%} of the background objects in the mask"

    # ...and the writing itself is untouched, which is the whole point of it being a veto.
    assert _covered(seeing, glyphs > 200) > 0.85, "the veto ate the text it was protecting"


def test_the_veto_stands_down_when_depth_does_not_describe_the_text():
    """Text too small for DepthCrafter to have responded to takes the depth behind it.

    A wall receding across the shot then hands every letter a different reading, and a veto
    run on that would bite the far end off the line. So a scattered vote has to mean "this
    crop cannot answer" rather than "keep whichever blob sat at the median".
    """
    frame, glyphs, _, box = _scene_with_lookalikes()
    ramp = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32), (H, 1))

    blind = _extract(frame, box)
    seeing = _extract(frame, box, depth=ramp)
    assert blind is not None and seeing is not None
    assert _covered(seeing, glyphs > 200) == pytest.approx(_covered(blind, glyphs > 200),
                                                           abs=0.01), \
        "a depth map that says nothing about the text must change nothing"


def test_the_full_stop_ending_a_subtitle_survives():
    """Found on real footage: every line ending in a full stop lost it.

    A depth map arrives blurred and often at a lower resolution than the picture, so a mark
    a twentieth the area of the letters beside it has no depth of its own to read - what
    comes back is the halo lying over it, which agrees with no letter. Small marks therefore
    sit the vote out rather than being judged on a reading that is not theirs.
    """
    from PIL import Image, ImageDraw

    base = np.full((H, W, 3), 45, np.uint8)
    font = load_font(52)
    img = Image.fromarray(base)
    ImageDraw.Draw(img).text((60, 120), "you should have come sooner.", font=font,
                             fill=(255, 255, 255))
    cover = Image.new("L", (W, H), 0)
    ImageDraw.Draw(cover).text((60, 120), "you should have come sooner.", font=font, fill=255)
    frame, cover = np.array(img), np.array(cover)

    # The mark on its own: everything past the last letter of "sooner".
    ys, xs = np.nonzero(cover > 200)
    stop = np.zeros_like(cover, bool)
    stop[:, xs.max() - 12:] = True
    stop &= cover > 200
    assert stop.sum() > 20, "test setup: the full stop was not isolated"

    box = (50, 110, xs.max() + 12, ys.max() + 12)
    # A low-resolution, heavily blurred map with a deep slab under the text - which is what
    # it takes to drag a small mark's reading far enough off the line to be rejected. The
    # first version of this used a milder guide and stopped being a test of anything: the
    # polarity fix and a wider tolerance made the mark safe on their own, so it passed with
    # the size guard removed.
    guide = _slab_guide(cover, text_depth=0.85, scene_depth=0.25, scale=0.18, sigma=6.0)
    patch = _extract(frame, box, depth=guide)
    assert patch is not None
    assert _covered(patch, stop) > 0.8, "the full stop was vetoed off the end of the line"
    letters = (cover > 200) & ~stop
    assert _covered(patch, letters) > 0.9, "and the letters must be untouched either way"


def test_the_tolerance_actually_controls_the_veto():
    """It silently stopped doing so, and nothing here noticed.

    The bar a blob is held to is the configured floor or a multiple of the blobs' own
    scatter, whichever is larger. That multiple was calibrated against a measurement taken
    while the polarity decision was inverting, so the "blobs" measured were the gaps between
    the glyphs rather than the glyphs - gaps are scattered across whatever the picture is
    doing, and needed a far wider bar than letters sitting on one slab. The multiple came out
    at 8x, which on ordinary text puts the scatter term permanently above the floor: every
    tolerance from 0.10 upwards produced a byte-identical mask and the knob did nothing.

    So this asks the only thing that matters about a control - that turning it changes the
    answer, monotonically, across the range it is offered over.

    The guide has to give the line a scatter of its own for that to mean anything. A slab
    with every letter on exactly the same value has a scatter of nearly zero, so the floor
    governs whatever the multiple is and the fixture passes with the bug still in it - which
    is how the original tests missed this.
    """
    frame, glyphs, objects, box = _scene_with_lookalikes()
    rng = np.random.default_rng(9)
    guide = np.full((H, W), 0.50, np.float32)
    halo = cv2.dilate((glyphs > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    guide[halo] = 0.80
    count, labels = cv2.connectedComponents(halo.astype(np.uint8))
    for label in range(1, count):
        guide[labels == label] += rng.normal(0, 0.025)  # letters wobble off the slab
    guide[objects > 0] = 0.60                           # and the intruder is a step away
    guide = cv2.GaussianBlur(guide, (0, 0), 1.5)

    covered = [_covered(_extract(frame, box, StrokeConfig(depth_tol=t), depth=guide),
                        objects > 0)
               for t in (0.05, 0.20, 0.60)]
    assert covered[0] < covered[-1], \
        f"the tolerance made no difference across its range: {covered}"
    assert covered == sorted(covered), f"raising it should never veto more: {covered}"


def test_depth_tol_of_zero_turns_the_veto_off():
    frame, glyphs, objects, box = _scene_with_lookalikes()
    guide = _slab_guide(glyphs)
    off = _extract(frame, box, StrokeConfig(depth_tol=0.0), depth=guide)
    assert _covered(off, objects > 0) > 0.5, "depth_tol=0 should leave the mask alone"


def test_a_slab_that_swallows_the_objects_too_keeps_them():
    """Honest about the limit: an object inside the corrupted halo shares the text's depth.

    DepthCrafter's smear is wider than the glyphs, so something sitting right against the
    writing is on the same slab and this test cannot see it. That is not a bug to paper
    over - it is the boundary of what depth can answer, and the reason the colour test and
    the temporal prior are still carrying their share.
    """
    frame, glyphs, objects, box = _scene_with_lookalikes()
    guide = _slab_guide(glyphs)
    guide[objects > 0] = 0.80  # pretend the smear reached them
    patch = _extract(frame, box, depth=cv2.GaussianBlur(guide, (0, 0), 1.5))
    assert _covered(patch, objects > 0) > 0.5


# -------------------------------------------------------------- the colour agreement veto

def test_an_object_of_another_colour_is_dropped_without_any_depth_map():
    """Weaker than depth and free: it needs nothing the extractor did not already have."""
    frame, glyphs, objects, box = _scene_with_lookalikes(colour=(255, 255, 90))
    kept = _extract(frame, box, StrokeConfig(chroma_tol=0.0))
    dropped = _extract(frame, box)
    assert kept is not None and dropped is not None
    assert _covered(kept, objects > 0) > 0.5, "the fixture is not reproducing the bug"
    assert _covered(dropped, objects > 0) < 0.05
    assert _covered(dropped, glyphs > 200) > 0.85


def test_the_colour_veto_leaves_ordinary_white_text_alone():
    """Every glyph of a credit is the same colour, so none of them should be near the gate."""
    frame, glyphs, _, box = _scene_with_lookalikes()
    strict = _extract(frame, box, StrokeConfig(chroma_tol=0.02))
    assert strict is not None
    assert _covered(strict, glyphs > 200) > 0.85


def test_neither_veto_fires_on_a_crop_with_too_few_blobs():
    """With two blobs there is no majority to be the odd one out of, so nothing is dropped."""
    from dsf.refine.strokes import _keep_agreeing

    blobs = [np.zeros((20, 20), bool) for _ in range(2)]
    blobs[0][2:8, 2:8] = True
    blobs[1][12:18, 12:18] = True
    values = np.zeros((20, 20), np.float32)
    values[12:18, 12:18] = 1.0  # wildly different, and still not enough to act on
    strength = np.ones((20, 20), np.float32)
    assert len(_keep_agreeing(blobs, values, 0.05, 0.6, strength)) == 2


# ----------------------------------------------------------------------------- the guide

def test_the_guide_normalises_over_the_legal_code_range():
    """So that depth_tol means the same fraction of the picture whatever the source tags."""
    from dsf.pipeline import depth_guide

    class Info:
        bit_depth, color_range = 10, "tv"

    cfg = PipelineConfig()
    plane = np.array([[64, 502, 940]], np.uint16)  # 10-bit tv: black, mid, white
    guide = depth_guide(plane, Info(), 3, 1, cfg)
    assert guide[0, 0] == pytest.approx(0.0, abs=1e-3)
    assert guide[0, 2] == pytest.approx(1.0, abs=1e-3)
    assert 0.4 < guide[0, 1] < 0.6


def test_the_guide_is_stretched_to_the_rgb_frame():
    """Depth maps routinely come back at another resolution, exactly as the masks do."""
    from dsf.pipeline import depth_guide

    class Info:
        bit_depth, color_range = 8, "pc"

    plane = np.zeros((90, 160), np.uint16)
    plane[45:, :] = 255
    guide = depth_guide(plane, Info(), 640, 360, PipelineConfig())
    assert guide.shape == (360, 640)
    assert guide[10, 10] == pytest.approx(0.0, abs=1e-3)
    assert guide[350, 10] == pytest.approx(1.0, abs=1e-3)


def test_pairing_the_streams_closes_both_readers():
    """`prefetch` closes the stream it owns when the consumer stops early, and on a clip
    that has seen enough it always does. A bare zip has no close, so both ffmpeg processes
    would be left to whenever the collector got round to them."""
    from dsf.pipeline import _with_guides

    closed: list[str] = []

    def reader(tag: str):
        try:
            for i in range(10):
                yield tag
        finally:
            closed.append(tag)

    paired = _with_guides(reader("rgb"), reader("depth"))
    next(paired)
    paired.close()
    assert sorted(closed) == ["depth", "rgb"]


def test_pairing_ends_with_the_shorter_stream():
    """Which is what check_alignment warns about when the two are different lengths."""
    from dsf.pipeline import _with_guides

    def reader(n: int):
        yield from range(n)

    assert len(list(_with_guides(reader(5), reader(3)))) == 3


def test_without_a_depth_map_every_frame_is_paired_with_nothing():
    from dsf.pipeline import _with_guides

    def reader(n: int):
        yield from range(n)

    assert list(_with_guides(reader(3), None)) == [(0, None), (1, None), (2, None)]


# ------------------------------------------------------------------------- the two streams

class _FixedBox:
    """A stand-in detector that reports one subtitle-shaped box on every frame, so every
    frame reaches the extractor and can report which depth frame arrived with it."""

    name = "fixed-box"

    def detect(self, frames):
        from dsf.detect.base import DetectorResult

        box = Detection(poly=bbox_to_poly(10, 40, 50, 52), score=1.0)
        return [DetectorResult(detections=[box]) for _ in frames]


def _numbered_pair(tmp_path, count=12, offset=0):
    """An RGB clip and a depth map whose Nth frame is a flat level encoding N."""
    from dsf.videoio import synth_rgb_video, synth_test_video

    w, h = 64, 64
    rgb = [np.full((h, w, 3), 40, np.uint8) for _ in range(count)]
    # 10-bit tv range: 64 is black, 940 white. One clear step per frame, and `offset` frames
    # of black padding in front so the two streams have to be told how to line up. The real
    # frames start a step above the padding, so frame 0 is distinguishable from it.
    levels = [64] * offset + [64 + 60 * (i + 1) for i in range(count)]
    depth = [np.full((h, w), lv, np.uint16) for lv in levels]

    rgb_path, depth_path = tmp_path / "rgb.mp4", tmp_path / "depth.mp4"
    synth_rgb_video(rgb_path, rgb, fps=24)
    synth_test_video(depth_path, depth, fps=24, lossless=True)
    return str(rgb_path), str(depth_path)


def _guides_seen(rgb_path, depth_path, monkeypatch, **kw):
    """The mean of the guide handed to the extractor, frame by frame."""
    import dataclasses

    import dsf.pipeline as pipeline
    from dsf.media import probe

    seen: list[float] = []

    def spy(frame, det, cfg, depth=None):
        seen.append(float("nan") if depth is None else float(depth.mean()))
        return None

    monkeypatch.setattr(pipeline, "extract_patch", spy)
    # The blank fixture has no text to be gated on; this is about which depth frame arrives.
    cfg = PipelineConfig()
    cfg = dataclasses.replace(cfg, filters=dataclasses.replace(cfg.filters,
                                                               scene_text="mask"))
    list(pipeline.iter_frame_items(rgb_path, cfg, probe(rgb_path),
                                   detectors=[_FixedBox()], depth_path=depth_path, **kw))
    return seen


@pytest.mark.slow
def test_each_frame_gets_its_own_depth_frame(tmp_path, monkeypatch):
    """An off-by-one between the two streams would silently veto against the wrong picture.

    Nothing downstream could catch it: the mask would still look like text, just occasionally
    missing a glyph the neighbouring frame's depth disagreed with.
    """
    rgb_path, depth_path = _numbered_pair(tmp_path)
    seen = _guides_seen(rgb_path, depth_path, monkeypatch)
    assert len(seen) == 12
    assert seen == sorted(seen), "the depth stream is not advancing with the RGB one"
    step = np.diff(seen)
    assert np.allclose(step, step[0], atol=0.01), f"frames drifted apart: {seen}"


@pytest.mark.slow
def test_depth_offset_realigns_a_padded_depth_map(tmp_path, monkeypatch):
    """The same flag `run_fix` already uses to line the pair up for compositing."""
    rgb_path, depth_path = _numbered_pair(tmp_path, offset=3)
    aligned = _guides_seen(rgb_path, depth_path, monkeypatch, depth_start=3)
    padded = _guides_seen(rgb_path, depth_path, monkeypatch)
    assert aligned[0] > padded[0], "--depth-offset did not skip the padding"
    assert aligned[:3] == pytest.approx(padded[3:6], abs=0.01)


@pytest.mark.slow
def test_seeking_moves_both_streams_together(tmp_path, monkeypatch):
    """What the preview does: jump to one frame deep in the clip and mask a window round it."""
    rgb_path, depth_path = _numbered_pair(tmp_path)
    whole = _guides_seen(rgb_path, depth_path, monkeypatch)
    sought = _guides_seen(rgb_path, depth_path, monkeypatch, seek_frame=5, max_frames=3)
    assert sought == pytest.approx(whole[5:8], abs=0.01)
