"""Ground truth for measuring masks: it has to be the letters, and nothing that merely looks like them.

Both failures this guards against are ones that actually happened while measuring a real
clip, and both were invisible in the score and obvious in the picture. A sunlit window is as
static and as amber as a static credit, so persistence labelled it as writing and every
number taken against that mask was wrong for an hour. And a shot that holds still cannot be
labelled by persistence at all, because the scenery is exactly as persistent as the text.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import load_font

_LABEL = Path(__file__).resolve().parents[1] / "scripts" / "label_glyphs.py"
_SCORE = Path(__file__).resolve().parents[1] / "scripts" / "score_masks.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


labeller = _load(_LABEL, "label_glyphs")
scorer = _load(_SCORE, "score_masks")

W, H = 640, 200


def _credit_frames(n=40, moving=True, on=range(10, 30)):
    """A warm scene with an amber credit burned in over part of it.

    With *moving* the background slides, which is the case persistence was built for; without
    it the background holds still, which is the case that defeats persistence and needs the
    text to leave the frame instead.
    """
    from PIL import Image, ImageDraw

    rng = np.random.default_rng(4)
    base = cv2.resize(rng.normal(0, 1, (H // 10, W // 10)).astype(np.float32), (W, H),
                      interpolation=cv2.INTER_CUBIC)
    base = (base - base.min()) / (np.ptp(base) + 1e-6)
    font = load_font(34)
    frames, cover = [], None
    for i in range(n):
        scene = np.roll(base, i * 6 if moving else 0, axis=1)
        # The credit's own hue throughout, with its brightest parts crossing the same
        # threshold the writing does. A scene that is not confusable with the text cannot
        # show what the two labelling methods do differently - but one that is confusable
        # *everywhere* is not the real case either, and cannot be labelled by anything: a
        # pixel whose background is already the text's colour looks the same either way.
        warm = np.stack([scene * 95 + 140, scene * 70 + 104, scene * 40 + 43], -1)
        img = Image.fromarray(np.clip(warm, 0, 255).astype(np.uint8))
        if i in on:
            ImageDraw.Draw(img).text((70, 90), "SOME NAME", font=font, fill=(255, 190, 80))
            if cover is None:
                c = Image.new("L", (W, H), 0)
                ImageDraw.Draw(c).text((70, 90), "SOME NAME", font=font, fill=255)
                cover = np.array(c) > 128
        frames.append(np.array(img))
    return frames, cover


def _write(tmp_path, frames, name="clip.mp4"):
    from dsf.videoio import synth_rgb_video

    path = tmp_path / name
    synth_rgb_video(path, frames, fps=24)
    return str(path)


def _args(**kw):
    base = dict(on=None, off=None, persistence=False, min_bright=210, min_sat=0.55,
                any_colour=False, min_on=0.7, min_gain=0.3, min_area=40, min_height=8,
                max_height=70, max_width=140)
    base.update(kw)
    return type("A", (), base)()


@pytest.mark.slow
def test_it_labels_the_letters_and_not_the_scene(tmp_path):
    frames, cover = _credit_frames(moving=True)
    got = labeller.build(_write(tmp_path, frames), _args())
    mask = got["mask"]
    inside = float(mask[cover].mean())
    outside = int((mask & ~cv2.dilate(cover.astype(np.uint8),
                                      np.ones((7, 7), np.uint8)).astype(bool)).sum())
    assert inside > 0.5, f"only {inside:.0%} of the credit was labelled"
    assert outside < 0.1 * int(mask.sum()), f"{outside} px labelled away from the text"


@pytest.mark.slow
def test_a_still_shot_is_labelled_by_the_text_leaving_rather_than_by_persistence(tmp_path):
    """The case that defeats persistence outright.

    Nothing moves here, so every warm thing in the frame is exactly as persistent as the
    writing. What still separates them is that the writing is not always there.
    """
    frames, cover = _credit_frames(moving=False)
    path = _write(tmp_path, frames)
    got = labeller.build(path, _args())
    assert got["method"] == "differencing", "should not have chosen persistence here"
    assert float(got["mask"][cover].mean()) > 0.5

    forced = labeller.build(path, _args(persistence=True))
    assert float(forced["mask"].sum()) > float(got["mask"].sum()) * 1.3, \
        "persistence on a still shot should sweep in scenery the differencing avoids"


def test_scenery_shaped_wrong_for_a_letter_is_dropped():
    """The sunlit window that got into a real mask: as static and as amber as the text."""
    m = np.zeros((200, 640), bool)
    m[90:130, 70:300] = False
    for x in range(70, 300, 30):
        m[95:125, x:x + 12] = True          # letters
    m[20:185, 400:470] = True               # a window: far too tall to be a glyph
    kept, n = labeller.letter_components(m, min_area=40, min_h=8, max_h=70, max_w=140)
    assert not kept[20:185, 400:470].any(), "the window survived the letter filter"
    assert n >= 6 and kept[95:125, 70:82].any(), "the letters did not"


def test_the_band_filter_drops_a_letter_shaped_thing_far_off_the_line():
    m = np.zeros((200, 640), bool)
    for x in range(70, 300, 30):
        m[95:125, x:x + 12] = True          # the line of text
    m[5:35, 500:512] = True                 # something letter-sized, nowhere near it
    out = labeller.in_text_band(m)
    assert out[95:125, 70:82].any()
    assert not out[5:35, 500:512].any()


def test_ranges_parse():
    assert labeller.parse_ranges("26-29") == [26, 27, 28, 29]
    assert labeller.parse_ranges("0-2,7") == [0, 1, 2, 7]


def test_scoring_separates_the_two_ways_a_mask_can_be_wrong():
    """Recall and precision are not interchangeable here, so they are reported apart.

    A mask that misses strokes leaves corrupted depth showing through the writing; one that
    overruns them is largely absorbed by the heal. A single figure would hide which of those
    a change had traded for the other.
    """
    truth = np.zeros((60, 200), bool)
    truth[20:40, 30:170] = True
    core = cv2.erode(truth.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)

    perfect = scorer.score(truth, truth, core)
    thin = scorer.score(truth & (np.arange(200)[None, :] < 100), truth, core)
    fat = scorer.score(cv2.dilate(truth.astype(np.uint8),
                                  np.ones((9, 9), np.uint8)).astype(bool), truth, core)
    assert perfect[0] > 0.99 and perfect[1] > 0.99 and perfect[2] > 0.99
    assert thin[1] < 0.6 and thin[2] > 0.95, "a thin mask loses recall, not precision"
    assert fat[1] > 0.99 and fat[2] < 0.7, "a fat mask loses precision, not recall"
