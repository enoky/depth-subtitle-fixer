"""Configuration objects and presets for the whole pipeline.

Every tunable knob lives here so the CLI, the Gradio UI and the mask-cache sidecar all
speak the same language.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def configure_model_cache(models_dir: Path | None = None) -> Path:
    """Point doctr/easyocr/HF weight caches at the project so it stays self-contained.

    Must be called *before* importing doctr or easyocr, because both read these
    environment variables at import time.
    """
    root = Path(models_dir) if models_dir else MODELS_DIR
    root.mkdir(parents=True, exist_ok=True)
    (root / "doctr").mkdir(exist_ok=True)
    (root / "easyocr").mkdir(exist_ok=True)
    os.environ.setdefault("DOCTR_CACHE_DIR", str(root / "doctr"))
    os.environ.setdefault("EASYOCR_MODULE_PATH", str(root / "easyocr"))
    os.environ.setdefault("HF_HOME", str(root / "hf"))
    # doctr imports defusedxml, which deprecation-warns on import. Nothing here can act on
    # it, and it would otherwise print on every single run.
    warnings.filterwarnings("ignore", message=r".*defusedxml\.cElementTree.*")
    return root


@dataclass(frozen=True)
class DetectConfig:
    """Which text detectors to run and how."""

    #: EasyOCR is available and unioned in when asked for, but it is not on by default: it
    #: costs about twice what docTR does and on the clips measured it found essentially the
    #: same glyphs - 99.9% of the same mask pixels for 1.9x the detection time. It reads
    #: scene text better, so it is worth adding back on a stylised title card; check the
    #: preview rather than paying for it on every render.
    detectors: tuple[str, ...] = ("doctr",)
    det_arch: str = "db_resnet50"
    min_score: float = 0.30
    batch_size: int = 4
    device: str = "auto"
    #: run detection on every Nth frame; masks are propagated in between
    detect_every: int = 1


@dataclass(frozen=True)
class FilterConfig:
    """Separates overlaid text (subtitles/credits) from text filmed in the scene."""

    #: "full" | "bottom:0.30" | "top:0.20" | "x0,y0,x1,y1" (normalised)
    roi: str = "bottom:0.30"
    min_text_height: float = 0.012
    max_text_height: float = 0.30
    max_aspect: float = 80.0
    #: "keep" leaves in-scene text alone (appearance + persistence gates active),
    #: "mask" masks anything the detectors find.
    scene_text: str = "keep"
    #: minimum P95-P50 luminance spread inside the box (0..1)
    min_contrast: float = 0.16
    #: maximum stddev of chroma inside the candidate glyph pixels (0..1)
    max_chroma_std: float = 0.14
    min_persist_frames: int = 3
    persist_window: int = 9
    persist_iou: float = 0.30
    #: allow tracks to move vertically between frames (scrolling credits)
    allow_vertical_scroll: bool = False


@dataclass(frozen=True)
class StrokeConfig:
    """Glyph-level alpha extraction inside each accepted detection box."""

    pad: int = 4
    #: "auto" | "light" (bright glyphs) | "dark" (dark glyphs)
    polarity: str = "auto"
    min_cc_area: int = 8
    max_cc_area_frac: float = 0.35
    min_stroke: float = 0.8
    #: absolute floor for the stroke-width cap, in px
    max_stroke: float = 16.0
    #: and as a fraction of the detected text height, so 4K credits are not rejected
    max_stroke_frac: float = 0.45
    #: background-median window as a fraction of text height. It has to be wide enough that
    #: the writing is a *minority* of the window - roughly a whole letter across. Sized off a
    #: stroke instead, a bold word wins its own median, its interior reads as background, and
    #: the word comes back hollow or vanishes outright.
    background_scale: float = 0.90
    #: minimum residual for a crop to contain text at all
    min_response: float = 0.05
    #: how strong a blob must be relative to the strongest text in the same crop. Text fades
    #: as a whole so its glyphs all measure ~1.0; scene detail amplified by a faint fade
    #: measures around half that.
    min_relative_strength: float = 0.75
    #: how much stronger one sign must be before magnitude alone decides polarity
    polarity_ratio: float = 1.6
    #: how far an outline pixel's luma may drift from the outline's own colour. Only the
    #: rim is colour-tested; the fill is found by opacity, which survives fades.
    luma_tol: float = 0.20
    #: containment gap needed before enclosure alone decides which sign is the text
    enclosure_margin: float = 0.15
    #: and how much must actually be contained for the test to count as evidence at all
    enclosure_min: float = 0.50
    #: px the mask grows from the glyph core into a hard drawn outline. Off by default:
    #: it is only safe when the outline really is hard-edged, and growing into the soft drop
    #: shadow that title cards usually carry leaves a speckled crust on every glyph.
    rim_expand: int = 0


@dataclass(frozen=True)
class TemporalConfig:
    """Flicker suppression across the alpha-mask stack, and the memory of where text was."""

    mode: str = "median"  # median | max | none
    window: int = 3

    #: Frames the prior looks over. Part-way through a fade the mask is normalised by
    #: whatever the text is showing at, so a small divisor amplifies the scene along with
    #: it. Nothing in a single frame separates a faint glyph from a lit edge behind it at
    #: that point - but the credit is also on screen at full strength a second later, and
    #: the lit edge never is. 0 disables the prior.
    prior_window: int = 21
    #: A frame counts as evidence once its text peaks at least this opaque.
    prior_min_level: float = 0.60
    #: Slack in px when matching a faint mark against the remembered shape.
    prior_tolerance: int = 3
    #: If the remembered shape explains less than this much of what a frame found overall,
    #: the text has moved and the prior stands down rather than erasing it - which is what
    #: keeps scrolling credits safe.
    prior_min_overlap: float = 0.50
    #: How much of a single mark the memory must explain for that mark to be kept. On static
    #: text the two populations are far apart - glyphs measure ~1.0, specks ~0.2 - so the
    #: exact value does not matter much, and it is set below `prior_min_overlap` on purpose:
    #: a frame whose text has drifted just far enough to sit near the stand-down bar keeps
    #: its glyphs whole rather than losing the ones that drifted furthest.
    prior_min_support: float = 0.35


@dataclass(frozen=True)
class CompositeConfig:
    """Depth repair: heal the corrupted halo, then paint the glyphs."""

    brightness: float = 0.92
    brightness_mode: str = "absolute"  # absolute | relative
    relative_offset: float = 0.08
    dilate: int = 2
    feather: float = 1.5
    heal: str = "edt"  # edt | none
    heal_scope: str = "glyph"  # glyph | region
    heal_dilate: int = 6
    heal_smooth: float = 2.0
    value_range: str = "auto"  # auto | tv | pc


@dataclass(frozen=True)
class EncodeConfig:
    encoder: str = "libx265"
    crf: int = 12
    preset: str = "slow"
    lossless: bool = False


@dataclass(frozen=True)
class PipelineConfig:
    detect: DetectConfig = field(default_factory=DetectConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    strokes: StrokeConfig = field(default_factory=StrokeConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    composite: CompositeConfig = field(default_factory=CompositeConfig)
    encode: EncodeConfig = field(default_factory=EncodeConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def lookahead(self) -> int:
        """Frames that must be buffered before the centre frame can be emitted."""
        return max(self.filters.persist_window, self.temporal.window)


#: Named presets applied before any explicit CLI flag.
PROFILES: dict[str, dict] = {
    "subtitles": {
        "filters": {"roi": "bottom:0.30", "min_persist_frames": 3, "allow_vertical_scroll": False},
        "detect": {"detect_every": 2},
        "temporal": {"mode": "median", "window": 3},
    },
    "credits": {
        "filters": {
            "roi": "full",
            "min_persist_frames": 2,
            "allow_vertical_scroll": True,
            "min_contrast": 0.12,
        },
        "detect": {"detect_every": 1},
        "temporal": {"mode": "max", "window": 3},
    },
    "both": {
        "filters": {"roi": "full", "min_persist_frames": 2, "allow_vertical_scroll": True},
        "detect": {"detect_every": 1},
        "temporal": {"mode": "median", "window": 3},
    },
}


def apply_profile(cfg: PipelineConfig, profile: str) -> PipelineConfig:
    """Return a copy of *cfg* with the named profile's overrides applied."""
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    updates = {}
    for section, values in PROFILES[profile].items():
        updates[section] = dataclasses.replace(getattr(cfg, section), **values)
    return dataclasses.replace(cfg, **updates)


def parse_roi(spec: str) -> tuple[float, float, float, float]:
    """Parse an ROI spec into normalised (x0, y0, x1, y1)."""
    spec = spec.strip().lower()
    if spec in ("full", "all", ""):
        return (0.0, 0.0, 1.0, 1.0)
    if ":" in spec:
        where, _, amount = spec.partition(":")
        frac = float(amount)
        if not 0.0 < frac <= 1.0:
            raise ValueError(f"ROI fraction must be in (0, 1]; got {frac}")
        if where == "bottom":
            return (0.0, 1.0 - frac, 1.0, 1.0)
        if where == "top":
            return (0.0, 0.0, 1.0, frac)
        raise ValueError(f"unknown ROI anchor {where!r}; use 'bottom' or 'top'")
    parts = [float(p) for p in spec.split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be 'full', 'bottom:F', 'top:F' or 'x0,y0,x1,y1'; got {spec!r}")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError(f"ROI box out of range or inverted: {parts}")
    return (x0, y0, x1, y1)
