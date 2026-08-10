"""Config precedence: a profile sets the baseline, explicit flags win."""

from __future__ import annotations

import os

import pytest

from dsf.cli import build_parser, parse_frame_list
from dsf.config import PipelineConfig, apply_profile, configure_model_cache


def cfg_from(argv):
    from dsf.cli import build_config

    return build_config(build_parser().parse_args(argv))


def test_profiles_differ_where_it_matters():
    subs = apply_profile(PipelineConfig(), "subtitles")
    credits = apply_profile(PipelineConfig(), "credits")

    assert subs.filters.roi == "bottom:0.30"
    assert credits.filters.roi == "full"
    assert not subs.filters.allow_vertical_scroll
    assert credits.filters.allow_vertical_scroll
    # Propagating a mask between frames is only sound for text that holds still.
    assert credits.detect.detect_every == 1
    # A subtitle sits over the far end of the shot, so a constant plane near the camera is
    # where it belongs. A credit lands anywhere - often over a face in the near field - and
    # a constant plane there can put the text further forward than the corruption it is
    # replacing, leaving the frame worse than untouched.
    assert subs.composite.brightness_mode == "absolute"
    assert credits.composite.brightness_mode == "relative"


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError):
        apply_profile(PipelineConfig(), "nope")


def test_explicit_flag_beats_the_profile():
    cfg = cfg_from(["fix", "--rgb", "a.mp4", "--depth", "b.mp4", "--out", "c.mp4",
                    "--profile", "subtitles", "--roi", "full"])
    assert cfg.filters.roi == "full"


def test_unset_flags_keep_the_profile_value():
    cfg = cfg_from(["fix", "--rgb", "a.mp4", "--depth", "b.mp4", "--out", "c.mp4",
                    "--profile", "credits"])
    assert cfg.filters.roi == "full"
    assert cfg.temporal.mode == "max"


def test_detectors_parse_as_a_tuple():
    cfg = cfg_from(["detect", "--rgb", "a.mp4", "--out-mask", "m.mkv",
                    "--detectors", "doctr,easyocr"])
    assert cfg.detect.detectors == ("doctr", "easyocr")


def test_only_doctr_runs_unless_easyocr_is_asked_for():
    """EasyOCR roughly doubles detection time for ~0.1% more mask on ordinary text.

    It is better at stylised scene text, so it stays available - but as something you turn
    on for a title card that needs it, not something every render pays for.
    """
    cfg = cfg_from(["fix", "--rgb", "a.mp4", "--depth", "b.mp4", "--out", "c.mp4"])
    assert cfg.detect.detectors == ("doctr",)


@pytest.mark.parametrize("profile", ["subtitles", "credits", "both"])
def test_no_profile_quietly_switches_easyocr_back_on(profile):
    cfg = cfg_from(["fix", "--rgb", "a.mp4", "--depth", "b.mp4", "--out", "c.mp4",
                    "--profile", profile])
    assert cfg.detect.detectors == ("doctr",)


def test_numeric_flag_only_applies_when_given():
    base = cfg_from(["fix", "--rgb", "a", "--depth", "b", "--out", "c"])
    assert base.strokes.min_response == pytest.approx(0.05)

    given = cfg_from(["fix", "--rgb", "a", "--depth", "b", "--out", "c",
                      "--min-response", "0.2"])
    assert given.strokes.min_response == pytest.approx(0.2)


def test_lossless_flag():
    base = cfg_from(["fix", "--rgb", "a", "--depth", "b", "--out", "c"])
    assert base.encode.lossless is False
    on = cfg_from(["fix", "--rgb", "a", "--depth", "b", "--out", "c", "--lossless"])
    assert on.encode.lossless is True


def test_lookahead_covers_both_windows():
    cfg = PipelineConfig()
    assert cfg.lookahead >= cfg.filters.persist_window
    assert cfg.lookahead >= cfg.temporal.window


@pytest.mark.parametrize("spec,expected", [
    ("12", [12]),
    ("12,40,100", [12, 40, 100]),
    ("0-4", [0, 1, 2, 3, 4]),
    ("0-10:5", [0, 5, 10]),
    ("100,0-2", [0, 1, 2, 100]),
])
def test_parse_frame_list(spec, expected):
    assert parse_frame_list(spec) == expected


def test_config_serialises_for_the_mask_sidecar():
    data = PipelineConfig().to_dict()
    assert data["composite"]["brightness"] == pytest.approx(0.92)
    assert "detectors" in data["detect"]


def test_model_cache_disables_doctrs_per_call_thread_pool(tmp_path, monkeypatch):
    """The scan leaks OS threads without this - see configure_model_cache for the mechanism."""
    from doctr.file_utils import ENV_VARS_TRUE_VALUES

    monkeypatch.delenv("DOCTR_MULTIPROCESSING_DISABLE", raising=False)
    configure_model_cache(tmp_path)
    assert os.environ["DOCTR_MULTIPROCESSING_DISABLE"].upper() in ENV_VARS_TRUE_VALUES


def test_model_cache_leaves_an_explicit_choice_alone(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCTR_MULTIPROCESSING_DISABLE", "FALSE")
    configure_model_cache(tmp_path)
    assert os.environ["DOCTR_MULTIPROCESSING_DISABLE"] == "FALSE"
