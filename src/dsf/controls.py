"""The single table of tunable controls the Gradio app is built from.

Declared once, in one place, so a knob cannot exist on the command line and quietly go
missing from the UI. ``tests/test_ui.py`` checks this table against the CLI's own argument
map, so adding a flag to one and forgetting the other fails the suite.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Callable

from .config import PipelineConfig, apply_profile

#: Layout groups, in the order they appear in the app.
GROUPS = ("detect", "detect_adv", "strokes_adv", "repair", "encode")


@dataclass(frozen=True)
class Knob:
    """One control: which config field it drives and how to draw it."""

    key: str
    section: str | None  # None for controls that are not config fields
    field: str | None
    group: str
    label: str
    kind: str  # slider | number | text | radio | checkbox | checkboxgroup
    #: kind-specific extras: choices, minimum, maximum, step, info
    options: dict = dataclasses.field(default_factory=dict)
    #: whether changing this needs the previewed frame redrawn
    live: bool = True

    def default(self, cfg: PipelineConfig) -> Any:
        if self.section is None:
            return self.options.get("value")
        value = getattr(getattr(cfg, self.section), self.field)
        return list(value) if isinstance(value, tuple) else value


def _k(key, section, field, group, label, kind, **options) -> Knob:
    live = options.pop("live", True)
    return Knob(key=key, section=section, field=field, group=group, label=label,
                kind=kind, options=options, live=live)


#: Every field the CLI can set, plus the handful of controls that only exist in the app.
KNOBS: list[Knob] = [
    # ---------------------------------------------------------------- detection (main)
    # Not preview-bound: switching profile rewrites every other control, and the app
    # redraws once after that has settled rather than once per control it touched.
    _k("profile", None, None, "detect", "Profile", "radio",
       choices=["subtitles", "credits", "both"], value="subtitles", live=False,
       info="preset defaults; the controls below then apply on top"),
    _k("detectors", "detect", "detectors", "detect", "Detectors", "checkboxgroup",
       choices=["doctr", "easyocr"]),
    _k("roi", "filters", "roi", "detect", "ROI", "text",
       info="full | bottom:0.30 | top:0.20 | x0,y0,x1,y1"),
    _k("scene_text", "filters", "scene_text", "detect", "Text filmed in the scene", "radio",
       choices=["keep", "mask"], info="keep leaves shop signs and the like alone"),
    _k("detect_every", "detect", "detect_every", "detect", "Detect every N frames", "slider",
       minimum=1, maximum=8, step=1),
    _k("min_persist_frames", "filters", "min_persist_frames", "detect",
       "Min persistence frames", "slider", minimum=1, maximum=10, step=1),

    # ------------------------------------------------------------ detection (advanced)
    _k("det_arch", "detect", "det_arch", "detect_adv", "docTR architecture", "radio",
       choices=["db_resnet50", "db_resnet34", "db_mobilenet_v3_large",
                "fast_base", "fast_small", "fast_tiny"]),
    _k("min_score", "detect", "min_score", "detect_adv", "Min detector confidence", "slider",
       minimum=0.0, maximum=1.0, step=0.01),
    _k("batch_size", "detect", "batch_size", "detect_adv", "Batch size", "slider",
       minimum=1, maximum=16, step=1),
    _k("device", "detect", "device", "detect_adv", "Device", "radio",
       choices=["auto", "cuda", "cpu"]),
    _k("min_text_height", "filters", "min_text_height", "detect_adv",
       "Min text height (fraction)", "slider", minimum=0.002, maximum=0.10, step=0.001),
    _k("max_text_height", "filters", "max_text_height", "detect_adv",
       "Max text height (fraction)", "slider", minimum=0.05, maximum=0.90, step=0.01),
    _k("min_contrast", "filters", "min_contrast", "detect_adv", "Min box contrast", "slider",
       minimum=0.0, maximum=0.60, step=0.01),
    _k("persist_window", "filters", "persist_window", "detect_adv", "Persistence window",
       "slider", minimum=1, maximum=21, step=2),

    # ------------------------------------------------- glyph extraction (advanced)
    _k("polarity", "strokes", "polarity", "strokes_adv", "Polarity", "radio",
       choices=["auto", "light", "dark"], info="override if the text reads inverted"),
    _k("background_scale", "strokes", "background_scale", "strokes_adv",
       "Background window (x text height)", "slider", minimum=0.2, maximum=2.0, step=0.05,
       info="raise it if bold or heavy text comes back hollow"),
    _k("min_response", "strokes", "min_response", "strokes_adv", "Min stroke contrast",
       "slider", minimum=0.005, maximum=0.30, step=0.005),
    _k("min_relative_strength", "strokes", "min_relative_strength", "strokes_adv",
       "Min strength vs strongest text", "slider", minimum=0.0, maximum=1.0, step=0.05,
       info="drops specks picked up from the scene part-way through a fade"),
    _k("solidify", "strokes", "solidify", "strokes_adv", "Solidify glyph bodies", "checkbox",
       info="fills bevelled or two-tone logotypes that come back half transparent"),
    _k("rim_expand", "strokes", "rim_expand", "strokes_adv", "Grow into outline (px)",
       "slider", minimum=0, maximum=10, step=1,
       info="only for hard drawn outlines; on shadowed text it speckles the glyphs"),
    _k("luma_tol", "strokes", "luma_tol", "strokes_adv", "Outline colour tolerance", "slider",
       minimum=0.02, maximum=0.60, step=0.01),
    _k("strokes_from", "strokes", "strokes_from", "strokes_adv", "Stroke shape from", "radio",
       choices=["luma", "hisam"],
       info="hisam uses a model trained on stroke masks; needs scripts/fetch_hisam.py, and "
            "costs about 0.25 s a frame"),
    _k("depth_strokes", "strokes", "depth_strokes", "strokes_adv", "Read strokes from depth",
       "checkbox",
       info="union the glyphs the depth map sees with the ones the picture shows - finds "
            "text the same colour as what it sits on, but can paint it fatter; check a "
            "preview"),
    _k("depth_tol", "strokes", "depth_tol", "strokes_adv", "Depth agreement tolerance",
       "slider", minimum=0.0, maximum=0.50, step=0.01,
       info="stops an object behind the text that matches its brightness being masked "
            "with it; 0 turns it off"),
    _k("chroma_tol", "strokes", "chroma_tol", "strokes_adv", "Colour agreement tolerance",
       "slider", minimum=0.0, maximum=0.50, step=0.01,
       info="the same test on hue, and it needs no depth map; 0 turns it off"),
    _k("cluster_min_agree", "strokes", "cluster_min_agree", "strokes_adv",
       "Agreement needed to reject", "slider", minimum=0.0, maximum=1.0, step=0.05,
       info="how much of a box must agree before the two tests above may drop the rest"),
    _k("max_stroke", "strokes", "max_stroke", "strokes_adv", "Max stroke width floor (px)",
       "slider", minimum=2, maximum=64, step=1),
    _k("min_cc_area", "strokes", "min_cc_area", "strokes_adv", "Min component area (px)",
       "slider", minimum=1, maximum=200, step=1),
    _k("pad", "strokes", "pad", "strokes_adv", "Box padding (px)", "slider",
       minimum=0, maximum=20, step=1),
    _k("temporal_mode", "temporal", "mode", "strokes_adv", "Temporal filter", "radio",
       choices=["median", "max", "none"]),
    _k("temporal_window", "temporal", "window", "strokes_adv", "Temporal window", "slider",
       minimum=1, maximum=15, step=2),
    _k("prior_window", "temporal", "prior_window", "strokes_adv",
       "Remember text over (frames)", "slider", minimum=0, maximum=61, step=2,
       info="stops a fade pulling scene detail in; 0 disables and speeds up previews"),
    _k("prior_min_level", "temporal", "prior_min_level", "strokes_adv",
       "Evidence needs opacity", "slider", minimum=0.1, maximum=1.0, step=0.05),

    # ------------------------------------------------------------------- depth repair
    _k("brightness", "composite", "brightness", "repair", "Brightness", "slider",
       minimum=0.0, maximum=1.0, step=0.005),
    _k("brightness_mode", "composite", "brightness_mode", "repair", "Brightness mode",
       "radio", choices=["absolute", "relative"]),
    _k("dilate", "composite", "dilate", "repair", "Dilate (px)", "slider",
       minimum=0, maximum=10, step=1),
    _k("feather", "composite", "feather", "repair", "Feather (sigma)", "slider",
       minimum=0.0, maximum=8.0, step=0.1),
    _k("heal", "composite", "heal", "repair", "Heal", "radio", choices=["edt", "none"]),
    _k("heal_scope", "composite", "heal_scope", "repair", "Heal scope", "radio",
       choices=["glyph", "region"]),
    _k("heal_strokes", "composite", "heal_strokes", "repair", "Heal halo (x stroke width)",
       "slider", minimum=0.0, maximum=6.0, step=0.1,
       info="what normally sets the healed radius; raise it if bad depth survives in a ring "
            "around the glyphs"),
    _k("heal_dilate", "composite", "heal_dilate", "repair", "Heal halo floor (px)", "slider",
       minimum=0, maximum=40, step=1,
       info="only takes over when it asks for more than the strokes imply"),
    _k("value_range", "composite", "value_range", "repair", "Luma code range", "radio",
       choices=["auto", "tv", "pc"]),

    # ----------------------------------------------------------------------- encoding
    # Output only - none of these change the previewed frame.
    _k("encoder", "encode", "encoder", "encode", "Encoder", "radio",
       choices=["libx265", "libx264", "ffv1"], live=False),
    _k("crf", "encode", "crf", "encode", "CRF (lower is better)", "slider",
       minimum=0, maximum=32, step=1, live=False),
    _k("preset", "encode", "preset", "encode", "Preset", "radio",
       choices=["ultrafast", "fast", "medium", "slow", "veryslow"], live=False),
    _k("lossless", "encode", "lossless", "encode", "Lossless", "checkbox", live=False),
]

KNOBS_BY_KEY = {k.key: k for k in KNOBS}
CONFIG_KNOBS = [k for k in KNOBS if k.section is not None]

_FIELD_TYPES = {
    (section, f.name): f.type
    for section in ("detect", "filters", "strokes", "temporal", "composite", "encode")
    for f in dataclasses.fields(getattr(PipelineConfig(), section))
}


def coerce(knob: Knob, value: Any) -> Any:
    """Turn a widget value into what the config field expects.

    Gradio hands back floats for sliders and lists for checkbox groups; the dataclasses want
    ints, tuples and bools. The declared type decides, so a new field needs no new code here.
    """
    declared = str(_FIELD_TYPES.get((knob.section, knob.field), "str"))
    if value is None:
        return None
    if declared.startswith("tuple"):
        return tuple(value)
    if declared == "bool":
        return bool(value)
    if declared == "int":
        return int(round(float(value)))
    if declared == "float":
        return float(value)
    return value


def build_config(values: dict[str, Any]) -> PipelineConfig:
    """Assemble a PipelineConfig from widget values.

    The profile sets the baseline and the controls apply on top - the same precedence the CLI
    uses, except that here every control always holds a value, so the app resets the controls
    to a profile's defaults whenever the profile changes (see `profile_defaults`).
    """
    cfg = apply_profile(PipelineConfig(), values.get("profile") or "subtitles")
    updates: dict[str, dict] = {}
    for knob in CONFIG_KNOBS:
        if knob.key not in values:
            continue
        coerced = coerce(knob, values[knob.key])
        if coerced is None:
            continue
        updates.setdefault(knob.section, {})[knob.field] = coerced
    for section, fields in updates.items():
        cfg = dataclasses.replace(cfg, **{section: dataclasses.replace(
            getattr(cfg, section), **fields)})
    return cfg


def profile_defaults(profile: str) -> dict[str, Any]:
    """What every control should read after the profile is switched."""
    cfg = apply_profile(PipelineConfig(), profile)
    return {k.key: k.default(cfg) for k in CONFIG_KNOBS}


def defaults() -> dict[str, Any]:
    cfg = apply_profile(PipelineConfig(), "subtitles")
    return {k.key: k.default(cfg) for k in KNOBS}


def make_component(gr, knob: Knob, value: Any):
    """Create the Gradio widget for one knob."""
    opts = dict(knob.options)
    opts.pop("value", None)
    info = opts.pop("info", None)
    common: dict[str, Any] = {"label": knob.label}
    if info:
        common["info"] = info
    if knob.kind == "slider":
        return gr.Slider(opts["minimum"], opts["maximum"], value=value,
                         step=opts.get("step", 1), **common)
    if knob.kind == "number":
        return gr.Number(value=value, **common)
    if knob.kind == "text":
        return gr.Textbox(value=value, **common)
    if knob.kind == "radio":
        return gr.Radio(opts["choices"], value=value, **common)
    if knob.kind == "checkbox":
        return gr.Checkbox(value=bool(value), **common)
    if knob.kind == "checkboxgroup":
        return gr.CheckboxGroup(opts["choices"], value=list(value or []), **common)
    raise ValueError(f"unknown control kind {knob.kind!r}")


#: Callable used by the app to lay groups out with a heading each.
GROUP_TITLES: dict[str, tuple[str, bool]] = {
    # group -> (title, collapsed-accordion?)
    "detect": ("Detection", False),
    "detect_adv": ("Detection — advanced", True),
    "strokes_adv": ("Glyph extraction — advanced", True),
    "repair": ("Depth repair", False),
    "encode": ("Encoding", True),
}


def group_knobs(group: str) -> list[Knob]:
    return [k for k in KNOBS if k.group == group]


ProfileFn = Callable[[str], dict]
