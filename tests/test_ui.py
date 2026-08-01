"""The Gradio app must at least construct - a broken control wiring is otherwise only
discovered when a user launches it."""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")


def _browse_buttons(demo):
    return [b for b in demo.blocks.values()
            if type(b).__name__ == "Button" and getattr(b, "value", None) == "Browse…"]


def test_app_builds_without_binding_a_port():
    from dsf.ui import Session, build_app

    demo = build_app(Session())
    assert demo is not None
    assert demo.title == "depth-subtitle-fixer"


def test_session_reports_nothing_loaded():
    from dsf.ui import Session

    session = Session()
    assert session.rgb_path is None
    assert session.max_frame() == 0


def test_browse_buttons_are_disabled_on_a_shared_link():
    """A dialog on a shared link opens on the server's desktop and blocks the viewer."""
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    buttons = _browse_buttons(demo)
    assert len(buttons) == 3, "one Browse button each for rgb, depth and output"
    assert all(b.interactive is False for b in buttons)


def test_browse_buttons_are_enabled_locally():
    from dsf.filedialog import dialogs_available
    from dsf.ui import Session, build_app

    if not dialogs_available():
        pytest.skip("no tkinter on this interpreter")
    demo = build_app(Session(), native_dialogs=True)
    buttons = _browse_buttons(demo)
    assert len(buttons) == 3
    assert all(b.interactive for b in buttons)


def test_every_cli_knob_is_reachable_in_the_ui():
    """The gap this guards against: 21 of 35 command-line knobs had no control at all,
    including the ones most often needed to rescue a difficult clip."""
    from dsf.cli import ARG_MAP
    from dsf.controls import CONFIG_KNOBS

    cli = {(section, field) for _, section, field in ARG_MAP}
    ui = {(k.section, k.field) for k in CONFIG_KNOBS}
    missing = sorted(cli - ui)
    assert not missing, f"exposed on the command line but not in the app: {missing}"


def test_the_ui_does_not_invent_settings_the_cli_lacks():
    """Drift in the other direction is just as confusing."""
    from dsf.cli import ARG_MAP
    from dsf.controls import CONFIG_KNOBS

    cli = {(section, field) for _, section, field in ARG_MAP}
    extra = sorted({(k.section, k.field) for k in CONFIG_KNOBS} - cli)
    assert not extra, f"in the app but not on the command line: {extra}"


def test_control_keys_are_unique():
    from dsf.controls import KNOBS

    keys = [k.key for k in KNOBS]
    assert len(keys) == len(set(keys))


def test_widget_values_are_coerced_to_the_config_types():
    """Gradio hands back floats for sliders and lists for checkbox groups."""
    from dsf.controls import build_config, defaults

    values = defaults()
    values.update({"detect_every": 3.0, "detectors": ["doctr"], "lossless": 1,
                   "crf": 14.0, "brightness": 0.5, "background_scale": 1.1})
    cfg = build_config(values)
    assert cfg.detect.detect_every == 3 and isinstance(cfg.detect.detect_every, int)
    assert cfg.detect.detectors == ("doctr",)
    assert cfg.encode.lossless is True
    assert cfg.encode.crf == 14 and isinstance(cfg.encode.crf, int)
    assert cfg.strokes.background_scale == pytest.approx(1.1)


def test_defaults_round_trip_to_the_pipeline_defaults():
    """Opening the app must not silently differ from running the CLI with no flags."""
    from dsf.config import PipelineConfig, apply_profile
    from dsf.controls import build_config, defaults

    assert build_config(defaults()) == apply_profile(PipelineConfig(), "subtitles")


def test_switching_profile_moves_the_controls():
    """Otherwise the preset is inert - every widget holds a value and overrides it."""
    from dsf.config import PipelineConfig, apply_profile
    from dsf.controls import build_config, defaults, profile_defaults

    values = defaults()
    values.update(profile_defaults("credits"))
    values["profile"] = "credits"
    assert build_config(values) == apply_profile(PipelineConfig(), "credits")
    assert values["roi"] == "full", "the credits profile searches the whole frame"


def test_encoding_controls_do_not_redraw_the_preview():
    """They only affect the written file, so they must not trigger a re-composite."""
    from dsf.controls import KNOBS

    encode = [k for k in KNOBS if k.group == "encode"]
    assert encode and all(not k.live for k in encode)
