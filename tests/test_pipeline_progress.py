"""Progress reporting: a preview frame is never one frame's work, and the app says so."""

from __future__ import annotations

import dataclasses

import pytest

from dsf.config import PipelineConfig, apply_profile
from dsf.pipeline import context_frames, context_radius


def test_a_single_preview_frame_costs_a_window_of_work():
    """The gates need the frames either side, so 'one frame' is misleading to a user."""
    cfg = apply_profile(PipelineConfig(), "subtitles")
    radius = context_radius(cfg)
    assert radius >= 1
    # Deep into a clip the window is symmetric and full width.
    assert context_frames(cfg, 500) == 2 * radius + 1
    # At the very start it is clipped to what exists.
    assert context_frames(cfg, 0) == radius + 1
    assert context_frames(cfg, 1) == radius + 2


@pytest.mark.parametrize("section,field", [
    ("filters", "persist_window"),
    ("temporal", "window"),
    ("temporal", "prior_window"),
])
def test_context_grows_with_every_gate_window(section, field):
    """Each of these looks either side of the frame, so each one has to widen the run."""
    base = PipelineConfig()
    wide = dataclasses.replace(base, **{section: dataclasses.replace(
        getattr(base, section), **{field: 41})})
    assert context_frames(wide, 500) > context_frames(base, 500)


def test_masks_for_frames_reports_progress(monkeypatch, tmp_path):
    """Without this the preview sits silent for a whole window of detections."""
    import numpy as np

    import dsf.pipeline as pipeline

    cfg = apply_profile(PipelineConfig(), "subtitles")
    seen: list[int] = []

    class FakeInfo:
        width, height, nb_frames = 8, 8, 50

    def fake_iter(rgb_path, cfg_, info, seek_frame=0, max_frames=None, detectors=None,
                  progress=None, **kw):
        for i in range(max_frames or 1):
            if progress:
                progress(i + 1)
            yield np.zeros((8, 8), np.uint8), []

    monkeypatch.setattr(pipeline, "iter_masks_detailed", fake_iter)
    pipeline.masks_for_frames("clip", cfg, [10], FakeInfo(),
                              progress=seen.append)
    assert seen, "no progress was reported while building a preview mask"
    assert seen == sorted(seen) and seen[-1] == context_frames(cfg, 10)
