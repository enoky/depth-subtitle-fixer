"""Loading a stroke model that lives outside the repository, and failing usefully without it.

Almost everything here runs whether or not Hi-SAM is installed, because most of what this
boundary owes the pipeline is behaviour when it is *absent*: a caller that asks for strokes
on a machine without 1.3 GB of weights should be told which script fetches them, not handed
an ImportError from three libraries down.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from dsf.refine import hisam


@pytest.fixture(autouse=True)
def _drop_cached_model():
    hisam.unload()
    yield
    hisam.unload()


def test_absence_is_reported_with_the_way_out(tmp_path, monkeypatch):
    """The message has to name the script, because nothing else in the tree hints at it."""
    monkeypatch.setattr(hisam, "CODE_DIR", tmp_path / "hi_sam")
    monkeypatch.setattr(hisam, "WEIGHTS_DIR", tmp_path / "hi_sam" / "pretrained_checkpoint")
    assert not hisam.available()

    with pytest.raises(hisam.Unavailable) as caught:
        hisam.strokes(np.zeros((8, 8, 3), np.uint8))
    message = str(caught.value)
    assert "fetch_hisam" in message, "the error must say what to run"
    assert str(tmp_path) in message, "and where it was looking"


def test_a_half_installed_copy_names_the_file_that_is_missing(tmp_path, monkeypatch):
    """Source without weights is the state `fetch_hisam.py` leaves behind on its own."""
    code = tmp_path / "hi_sam" / "hi_sam" / "modeling"
    code.mkdir(parents=True)
    (code / "build.py").write_text("", encoding="utf-8")
    weights = tmp_path / "hi_sam" / "pretrained_checkpoint"
    weights.mkdir(parents=True)
    monkeypatch.setattr(hisam, "CODE_DIR", tmp_path / "hi_sam")
    monkeypatch.setattr(hisam, "WEIGHTS_DIR", weights)

    assert not hisam.available()
    with pytest.raises(hisam.Unavailable) as caught:
        hisam.strokes(np.zeros((8, 8, 3), np.uint8))
    assert hisam.DEFAULT_HEAD in str(caught.value)


def test_the_model_is_built_once_and_kept(monkeypatch):
    """A ViT-L is 1.2 GB and seconds to construct, against 0.25 s to run a frame."""
    built = []

    def fake_build(head, model_type, device):
        built.append((head, model_type, device))
        return object()

    monkeypatch.setattr(hisam, "_build", fake_build)
    first = hisam.predictor()
    second = hisam.predictor()
    assert first is second and len(built) == 1

    # ...but a different head is a different model, and must not silently return the old one.
    other = hisam.predictor(head="something_else.pth")
    assert other is not first and len(built) == 2


def test_the_mask_comes_back_as_a_float_shape_not_a_tensor(monkeypatch):
    """What the extractor consumes is a float32 in [0, 1], whatever the model returned.

    The model answers with logits on whichever device it ran on. Handing those downstream
    would put a CUDA tensor into numpy arithmetic that has never seen one.
    """
    class FakePredictor:
        def set_image(self, image):
            self.shape = image.shape[:2]

        def predict(self, multimask_output=False):
            logits = np.full((1, *self.shape), -3.0, np.float32)
            logits[0, 2:5, 2:5] = 4.0
            return None, logits, None, None

    monkeypatch.setattr(hisam, "_build", lambda *a: FakePredictor())
    out = hisam.strokes(np.zeros((8, 12, 3), np.uint8))
    assert out.shape == (8, 12) and out.dtype == np.float32
    assert set(np.unique(out)) <= {0.0, 1.0}
    assert out[3, 3] == 1.0 and out[0, 0] == 0.0, "the sign of the logit decides"


def test_unload_lets_the_memory_go(monkeypatch):
    monkeypatch.setattr(hisam, "_build", lambda *a: object())
    first = hisam.predictor()
    hisam.unload()
    assert hisam.predictor() is not first


def test_a_segmented_word_is_not_thrown_out_for_being_too_big():
    """The filters that ask "too big to be a letter" are about what a residual gets wrong.

    A residual answers to anything thin and contrasty, so a slab filling the crop is a lit
    wall it mistook for writing, and the cap at 35% of the crop is what removes it. A model
    trained on stroke masks has already settled that question, and its strokes come out a
    little fatter than the letters, which on tightly-set text joins them up. Measured on a
    real credit, a whole word arrived as one 9,649 px component where the residual had given
    fourteen letters, the cap threw the word away, and the mask came out at 0.447 fgIoU
    against the 0.817 the model had actually produced.
    """
    from dsf.config import StrokeConfig
    from dsf.refine.strokes import _filter_components, _stroke_width, stroke_bounds

    # Letters joined along a baseline, which is what merging looks like - not a slab. A slab
    # would be rejected for its stroke width instead, and would prove nothing about the cap.
    crop = np.zeros((80, 260), np.uint8)
    crop[52:70, 20:240] = 1
    for x in range(22, 236, 26):
        crop[22:52, x:x + 16] = 1
    cfg = StrokeConfig()
    assert int(crop.sum()) > cfg.max_cc_area_frac * crop.size, "test setup: not over the cap"
    lo, hi = stroke_bounds(cfg, 70)
    assert lo <= _stroke_width(crop > 0) <= hi, "test setup: rejected for thickness, not size"

    residual = _filter_components(crop, cfg, text_height=70, segmented=False)
    model = _filter_components(crop, cfg, text_height=70, segmented=True)
    assert residual is None, "the cap should still remove a slab a residual found"
    assert model is not None and float(model.mean()) > 0.3, \
        "a word a stroke model segmented must survive its own size"


@pytest.mark.slow
def test_it_runs_without_moving_the_working_directory():
    """The upstream builder finds SAM through a path relative to the process's cwd.

    That is why its own demo has to be run from its own folder, and it is the one thing this
    module cannot inherit: frames are decoded on a worker thread, and `chdir` is process-wide,
    so a load would move the ground under a decode running beside it.
    """
    import os

    if not hisam.available():
        pytest.skip("Hi-SAM not fetched; run scripts/fetch_hisam.py")
    before = os.getcwd()
    frame = np.zeros((256, 512, 3), np.uint8)
    frame[120:140, 60:300] = 240
    mask = hisam.strokes(frame, device="cpu")
    assert os.getcwd() == before
    assert mask.shape == frame.shape[:2] and mask.dtype == np.float32
