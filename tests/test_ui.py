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


def _deps(demo):
    return list(demo.fns.values())


def _named(demo, name):
    return [f for f in _deps(demo) if getattr(f.fn, "__name__", "") == name]


def _direct(fns):
    """Bindings to a real control, as opposed to a `.then` chained off another event."""
    return [f for f in fns if any(cid is not None for cid, _ in f.targets)]


def _chained(fns):
    return [f for f in fns if all(cid is None for cid, _ in f.targets)]


def test_the_preview_listens_to_user_input_not_to_value_changes():
    """`change` also fires when a value is set programmatically.

    Switching profile rewrites every other control, so a `change` binding kicked off one
    redraw per control it touched - dozens of them, each recompositing the frame, before
    anything appeared.
    """
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    direct = _direct(_named(demo, "render_frame"))
    assert len(direct) > 20, "expected the preview bound to every live control"
    events = {event for fn in direct for _, event in fn.targets}
    assert events == {"input"}, f"preview redraws bound to {sorted(events)}"


def test_dragging_a_control_coalesces_instead_of_queueing():
    """Without this a drag queues a full recomposite per step and runs long after you stop."""
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    modes = {fn.trigger_mode for fn in _direct(_named(demo, "render_frame"))}
    assert modes == {"always_last"}, f"preview trigger modes: {modes}"


def test_switching_profile_redraws_exactly_once():
    """One `on_profile` to rewrite the controls, with a single redraw chained after it."""
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    profile_events = _named(demo, "on_profile")
    assert len(profile_events) == 1
    assert [event for _, event in profile_events[0].targets] == ["change"], \
        "the reset itself must run on change - input would miss a programmatic profile set"

    # The redraw is chained off the reset rather than bound to the profile control, so it
    # runs after the other controls have settled.
    profile_id = profile_events[0].targets[0][0]
    renders = _named(demo, "render_frame")
    bound_to_profile = [fn for fn in renders
                        if any(cid == profile_id for cid, _ in fn.targets)]
    assert not bound_to_profile, "profile must not also trigger a redraw directly"
    assert len(_chained(renders)) == 1, "expected exactly one redraw chained after the reset"


def test_the_render_result_is_reported_not_handed_back_as_a_file():
    """Gradio serves a file by copying it into its own cache, and refuses outright for
    anything outside the working directory or the temp folder.

    Rendering to a normal location - which is most of them - therefore blew up *after* the
    depth map had been written, so a finished render looked like a crash. Reporting the path
    also avoids duplicating gigabytes to hand back a file the user chose the location of.
    """
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    renders = _named(demo, "render_clip")
    assert len(renders) == 1
    outputs = renders[0].outputs
    assert len(outputs) == 1, "the status message should be the only output"

    kinds = {type(block).__name__ for block in demo.blocks.values()}
    assert "File" not in kinds, "a File output would reintroduce the cache-copy failure"


def test_load_always_repoints_the_output_at_the_pair_just_loaded(tmp_path):
    """Otherwise the previous clip's path survives into the next one.

    It renders to the wrong place under a name claiming to be the reel you just finished,
    and the first sign of it is an overwritten file.
    """
    from dsf.filedialog import suggest_output
    from dsf.ui import Session, build_app

    session = Session()
    session.load = lambda rgb, depth: "loaded"
    rgb, depth = tmp_path / "reel2.mp4", tmp_path / "reel2_depth.mp4"
    for path in (rgb, depth):
        path.write_bytes(b"")

    demo = build_app(session, native_dialogs=False)
    on_load = _named(demo, "on_load")[0].fn

    stale = str(tmp_path / "reel1_depth_fixed.mp4")
    _, _, out = on_load(str(rgb), str(depth), stale)
    assert out == suggest_output(str(depth))
    assert out != stale


def test_a_half_filled_load_leaves_the_output_box_alone(tmp_path):
    """Nothing was loaded, so there is nothing to name the output after."""
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    on_load = _named(demo, "on_load")[0].fn
    typed = str(tmp_path / "chosen.mp4")
    assert on_load("", "", typed)[2] == typed


def test_a_failed_render_reports_instead_of_raising(monkeypatch, tmp_path):
    """A bad render should say so in the app rather than surfacing a raw traceback."""
    import dsf.pipeline as pipeline
    from dsf.controls import KNOBS, defaults
    from dsf.ui import Session, build_app

    session = Session()
    session.rgb_path, session.depth_path = str(tmp_path), str(tmp_path)

    def explode(*a, **k):
        raise RuntimeError("codec went missing")

    # render_clip imports run_fix at call time, so patching the module reaches it.
    monkeypatch.setattr(pipeline, "run_fix", explode)

    demo = build_app(session, native_dialogs=False)
    render_clip = _named(demo, "render_clip")[0].fn
    values = defaults()
    message = render_clip(str(tmp_path / "out"), 0,
                          *[values[k.key] for k in KNOBS])
    assert "Render failed" in message and "codec went missing" in message


def test_the_render_status_sits_above_the_encoding_panel():
    """Gradio draws the progress bar over the output component, so where that component
    lives decides whether the bar is visible.

    Below the Encoding panel it lands off the bottom of a tall page - the bar was updating
    correctly and was still no use, which is indistinguishable from having none.
    """
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    blocks = list(demo.blocks.values())

    status = next(b for b in blocks
                  if type(b).__name__ == "Markdown"
                  and "Ready to render" in str(getattr(b, "value", "")))
    encoder = next(b for b in blocks
                   if type(b).__name__ == "Radio"
                   and list(getattr(b, "choices", []) or []) and
                   any("libx265" in str(c) for c in b.choices))
    # Components are numbered in creation order, so this is layout order.
    assert status._id < encoder._id, \
        "the render status must be laid out before the Encoding panel, not after it"


def test_the_render_status_is_never_empty():
    """An empty Markdown collapses to no height, leaving the bar nowhere to draw."""
    from dsf.ui import Session, build_app

    demo = build_app(Session(), native_dialogs=False)
    status = [b for b in demo.blocks.values()
              if type(b).__name__ == "Markdown" and "Ready to render" in str(getattr(b, "value", ""))]
    assert status, "the status component should start with placeholder text"
