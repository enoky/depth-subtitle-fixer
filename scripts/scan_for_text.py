"""Triage a folder of RGB clips: which ones carry burned-in subtitles or overlaid credits?

`dsf` already knows how to tell an overlay from a shop sign - that judgement lives in the
ROI/appearance/persistence gates in `dsf.filters` and the stroke test in
`dsf.refine.strokes`, and `dsf.pipeline.iter_masks_detailed` hands back only the detections
that survived all of it. So this tool detects nothing itself. It runs that pipeline over a
sample of each clip and asks one question of the result: did real overlay text show up often
enough, and for long enough in a row, to count?

Clips that pass are copied to `<output>/rgb_with_text/`, and their depth maps - same name
with `_depth` on the stem - to `<output>/depth_with_text/`.

    .venv/Scripts/python scripts/scan_for_text.py

The scan runs on one worker thread that never touches a widget; it posts messages to a queue
the main thread drains on a timer. Tk is not thread-safe, and a scan is minutes long.
"""

from __future__ import annotations

import contextlib
import csv
import json
import queue
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

# Run from a checkout without installing the package.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402

from dsf.config import PipelineConfig, apply_profile, configure_model_cache  # noqa: E402
from dsf.detect.base import DetectorResult, merge_detections  # noqa: E402
from dsf.filedialog import VIDEO_FILETYPES  # noqa: E402
from dsf.prefetch import prefetch  # noqa: E402
from dsf.sequence import IMAGE_SUFFIXES, is_sequence  # noqa: E402

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:  # a headless or stripped Python; main() explains it
    tk = None


PROFILES = ("subtitles", "credits", "both")
DEPTH_SUFFIX = "_depth"
REPORT_NAME = "scan_report.csv"
RGB_OUT = "rgb_with_text"
DEPTH_OUT = "depth_with_text"
SETTINGS_PATH = Path.home() / ".dsf" / "scan_for_text.json"


def video_suffixes() -> tuple[str, ...]:
    """The container extensions the app already offers in its file dialogs.

    Taken from `dsf.filedialog` rather than restated here, so a format added there is
    picked up by the scanner without anyone remembering to.
    """
    for label, patterns in VIDEO_FILETYPES:
        if label == "Video files":
            return tuple(sorted({p.lstrip("*").lower() for p in patterns.split()}))
    raise RuntimeError("dsf.filedialog.VIDEO_FILETYPES lost its 'Video files' row")


VIDEO_SUFFIXES = video_suffixes()
MEDIA_SUFFIXES = tuple(sorted(set(VIDEO_SUFFIXES) | set(IMAGE_SUFFIXES)))


# --------------------------------------------------------------------------- clips

@dataclass(frozen=True)
class Clip:
    """One thing to scan: a movie file, or a folder of stills."""

    path: Path
    sequence: bool = False

    @property
    def key(self) -> str:
        """The name a depth map would be built from - a folder has no extension to strip."""
        return (self.path.name if self.sequence else self.path.stem).lower()

    @property
    def label(self) -> str:
        return self.path.name


def _sorted_entries(folder: Path) -> list[Path]:
    try:
        return sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []


def discover_clips(root: str | Path, recursive: bool = False,
                   skip: Iterable[str | Path] = ()) -> list[Clip]:
    """Find every clip under *root*.

    A folder of stills is a clip, not a folder to descend into - which is why the sequence
    test comes before the recursion. *skip* keeps the scan out of its own output folder when
    the user points both at the same place.
    """
    root = Path(root)
    blocked = set()
    for path in skip:
        if not path:
            continue
        try:
            blocked.add(Path(path).resolve())
        except OSError:
            continue
    found: list[Clip] = []

    def walk(folder: Path) -> None:
        for entry in _sorted_entries(folder):
            try:
                if entry.resolve() in blocked:
                    continue
            except OSError:
                continue
            if entry.is_dir():
                if is_sequence(entry):
                    found.append(Clip(entry, sequence=True))
                elif recursive:
                    walk(entry)
            elif entry.suffix.lower() in VIDEO_SUFFIXES:
                found.append(Clip(entry))

    walk(root)
    return found


# --------------------------------------------------------------------------- depth pairing

def index_depth(root: str | Path, recursive: bool = True) -> dict[str, list[Path]]:
    """Map every depth map under *root* to the stem it would pair with.

    Built once for the whole run: a per-clip directory search would restat the depth folder
    for every clip in the scan.
    """
    index: dict[str, list[Path]] = {}

    def add(key: str, path: Path) -> None:
        index.setdefault(key.lower(), []).append(path)

    def walk(folder: Path) -> None:
        for entry in _sorted_entries(folder):
            if entry.is_dir():
                if is_sequence(entry):
                    add(entry.name, entry)
                elif recursive:
                    walk(entry)
            elif entry.suffix.lower() in MEDIA_SUFFIXES:
                add(entry.stem, entry)

    root = Path(root)
    if root.is_dir():
        walk(root)
    return index


def _depth_preference(clip: Clip) -> Callable[[Path], tuple]:
    """Rank candidates when a depth folder offers more than one match for a name."""
    def key(path: Path) -> tuple:
        # A sequence clip wants a sequence depth map and vice versa; then prefer a video
        # over a loose still; then name order, so the choice is at least deterministic.
        shape_match = 0 if path.is_dir() == clip.sequence else 1
        kind = 0 if path.suffix.lower() in VIDEO_SUFFIXES or path.is_dir() else 1
        return (shape_match, kind, path.name.lower())
    return key


def find_depth(clip: Clip, index: dict[str, list[Path]]) -> tuple[Path | None, str]:
    """Locate *clip*'s depth map. Returns ``(path, status)``.

    The extension is deliberately not required to match: a PNG-sequence RGB clip is often
    paired with an mkv depth map, and vice versa.
    """
    matches = index.get(f"{clip.key}{DEPTH_SUFFIX}", [])
    if not matches:
        return None, "missing"
    if len(matches) == 1:
        return matches[0], "found"
    return sorted(matches, key=_depth_preference(clip))[0], "ambiguous"


# --------------------------------------------------------------------------- sampling plan

def plan_windows(total_frames: int, windows: int, window_len: int, radius: int,
                 exhaustive: bool = False) -> list[tuple[int, int | None]]:
    """Where to sample, as ``(seek_frame, max_frames)`` pairs.

    Windows of *consecutive* frames, never scattered single ones. The persistence gate is
    what rejects a shop sign, and it decides by looking at a detection's neighbours in time;
    hand it isolated frames and it either sees an empty window or is skipped entirely, and
    the scene-text rejection this whole tool depends on quietly stops working. So a window
    is never shorter than the context the gates need either side of a frame.

    A frame count of 0 means the container would not say - fall back to reading the clip
    through, which the early exit makes cheap on anything that does have text.
    """
    window_len = max(int(window_len), 2 * max(0, int(radius)) + 1)
    windows = max(1, int(windows))
    if exhaustive or total_frames <= 0:
        return [(0, None)]
    if windows == 1 or total_frames <= window_len * windows:
        # Sampling would cost as much as reading the lot, so read the lot.
        return [(0, int(total_frames))]
    span = total_frames - window_len
    starts = sorted({int(round(i * span / (windows - 1))) for i in range(windows)})
    return [(s, window_len) for s in starts]


# --------------------------------------------------------------------------- the verdict

@dataclass(frozen=True)
class Sweep:
    """The cheap first look: is there anything text-shaped in this clip at all?

    Detection alone - no stroke extraction, no temporal filter, no prior. Those cost about
    twice what detection does and they exist to decide *which pixels* are text, a question
    worth asking only once something has been found. On a folder where most clips are clean
    this is where nearly all the time is saved, because a clean clip never gets to stop
    early: it pays for every frame it is given.

    Sound as a pre-filter because it applies a strict subset of the confirm pass's gates to
    the same detector, so anything it does not see, the confirm pass would not have seen on
    those frames either.
    """

    #: Seek points spread across the clip. A seek costs ~250ms of ffmpeg startup - about ten
    #: frames of detection - so a few long clusters beat many short ones. At one frame per
    #: seek, startup is 90% of the bill.
    #: Deliberately the same count as the confirm pass's blind windows: the sweep looks in
    #: the same *places*, just far less deeply at each (six frames instead of forty-five).
    #: Text that appears only briefly in a long clip can be missed by any sampled scan, and
    #: this keeps that risk exactly where it already was rather than making it worse. Raise
    #: it for long clips with brief titles; each extra cluster costs one ffmpeg startup.
    clusters: int = 8
    #: Consecutive frames read at each one.
    frames: int = 6
    #: Frames within a cluster that must carry a candidate before the clip is escalated.
    #: Real overlay text sits still for a second or more, so it lands on every frame of a
    #: cluster; a detector twitching once does not.
    min_cluster_hits: int = 2
    #: Side length docTR resizes to for the sweep. Detection is dominated by this, not by
    #: the clip's own resolution: 640 runs 1.7x faster than the 1024 default and still finds
    #: 85% of the boxes, and on measurement it produced no boxes at all on clean footage at
    #: any size - dropping it costs recall, never precision. The confirm pass gets the full
    #: 1024 because it needs accurate boxes to extract strokes from; the sweep only needs to
    #: notice that writing exists. 0 leaves the detector alone.
    input_size: int = 640


@dataclass(frozen=True)
class Reading:
    """The last look: does the confirmed text actually read as words?

    The gates upstream ask whether something was *overlaid* rather than filmed, which is a
    question about how a region behaves, not about what it says - so a railing, a window
    grid or a run of compression noise that happens to sit still and hold its colour can
    walk through all of them. Handing the region to a recogniser asks the one question that
    separates writing from structure.

    Latin script only (crnn_vgg16_bn), so a clip subtitled in Chinese, Japanese, Korean,
    Cyrillic or Arabic will not read - turn the requirement off for that footage. The words
    are recorded in the report either way, which is what makes a false positive diagnosable
    instead of mysterious.
    """

    require: bool = True
    #: Alphanumeric characters a word needs before it counts. A single confident character
    #: is what a false box reads as; a subtitle is a sentence.
    min_chars: int = 3
    min_confidence: float = 0.45
    #: Frames to try, best evidence first, before giving up on a clip.
    max_frames: int = 3


@dataclass(frozen=True)
class Thresholds:
    """What it takes for a clip to count as carrying overlay text."""

    #: Fraction of the frame the mask must cover, weighted by opacity. One stray blob on one
    #: frame is not a subtitle, and without a floor here it would be treated as one.
    min_coverage: float = 2e-5
    #: Text frames needed in total.
    min_text_frames: int = 6
    #: ...of which this many must be consecutive. Overlays stay put for a second or more;
    #: a detector twitching at a passing highlight does not.
    min_run: int = 3

    def sane(self) -> "Thresholds":
        return Thresholds(max(0.0, float(self.min_coverage)),
                          max(1, int(self.min_text_frames)),
                          max(1, int(self.min_run)))


class Verdict:
    """Accumulates per-frame evidence and answers 'enough yet?'."""

    def __init__(self, thresholds: Thresholds):
        self.t = thresholds.sane()
        self.frames_scanned = 0
        self.text_frames = 0
        self.longest_run = 0
        self.peak_coverage = 0.0
        self.hits: list[int] = []
        self._run = 0

    def start_window(self) -> None:
        """A new window is not continuous with the last one, so the run restarts."""
        self._run = 0

    def observe(self, index: int, coverage: float, detections: int) -> bool:
        """Record one frame. Returns True once the clip has met the bar."""
        self.frames_scanned += 1
        self.peak_coverage = max(self.peak_coverage, float(coverage))
        if detections >= 1 and coverage >= self.t.min_coverage:
            self.text_frames += 1
            self._run += 1
            self.longest_run = max(self.longest_run, self._run)
            if len(self.hits) < 8:
                self.hits.append(int(index))
        else:
            self._run = 0
        return self.flagged

    @property
    def flagged(self) -> bool:
        return (self.text_frames >= self.t.min_text_frames
                and self.longest_run >= self.t.min_run)

    def evidence(self) -> str:
        return (f"{self.text_frames}/{self.frames_scanned} frames, run {self.longest_run}, "
                f"peak {self.peak_coverage:.2e}")


@dataclass
class ClipResult:
    clip: Clip
    verdict: str = "clean"  # text | rejected | clean | error | skipped | cancelled
    #: How far the clip got: swept (nothing text-shaped) | confirmed (overlay gates) |
    #: read (recognised as words). Says which stage made the call.
    stage: str = "swept"
    swept_frames: int = 0
    swept_of: int = 0
    sweep_hits: int = 0
    frames_scanned: int = 0
    text_frames: int = 0
    longest_run: int = 0
    peak_coverage: float = 0.0
    hits: list[int] = field(default_factory=list)
    words: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    error: str = ""

    @property
    def evidence(self) -> str:
        if self.verdict in ("error", "skipped"):
            return self.error
        if self.stage == "swept":
            of = f" of {self.swept_of}" if self.swept_of else ""
            return f"nothing text-shaped in {self.swept_frames}{of} swept frames"
        detail = (f"{self.text_frames}/{self.frames_scanned} frames, run {self.longest_run}, "
                  f"peak {self.peak_coverage:.2e}")
        if self.words:
            detail += f" | read {' '.join(self.words[:4])!r}"
        elif self.verdict == "rejected":
            detail += " | did not read as words"
        return detail


# --------------------------------------------------------------------------- scanning

def build_scan_config(profile: str, detect_every: int = 0, batch_size: int = 4,
                      device: str = "auto", use_easyocr: bool = False) -> PipelineConfig:
    """Profile first, then the handful of knobs the scanner exposes - as the CLI does.

    `filters.scene_text` is left at "keep" on purpose and is not exposed anywhere in the UI.
    It is the switch that turns the ROI, appearance and persistence gates on, and those
    gates are the entire reason a licence plate or a shop sign does not flag a clip.
    """
    cfg = apply_profile(PipelineConfig(), profile)
    updates: dict = {"batch_size": max(1, int(batch_size)), "device": device or "auto"}
    if detect_every and int(detect_every) > 0:
        updates["detect_every"] = int(detect_every)
    if use_easyocr:
        updates["detectors"] = ("doctr", "easyocr")
    cfg = replace(cfg, detect=replace(cfg.detect, **updates))
    assert cfg.filters.scene_text == "keep"
    return cfg


def _batches(frames: Iterable, size: int) -> Iterator[list]:
    """Group a frame stream without ever holding more than one batch of it in memory."""
    batch: list = []
    for frame in frames:
        batch.append(frame)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


#: Clip workers. Each is mostly waiting - on ffmpeg startup, on decoding, on the disk - and
#: they share one locked detector, so a second worker fills the gaps a first one leaves
#: rather than competing for the GPU. Past two or three the GPU is saturated and more
#: workers only add memory.
DEFAULT_WORKERS = 3


#: Most of a clip the sweep may look at. Past this it is paying ffmpeg startups to re-read
#: what the confirm pass would have read in one go, and there is nothing left to save.
SWEEP_MAX_COVERAGE = 0.25


def sweep_plan(total_frames: int, sweep: Sweep) -> list[tuple[int, int]]:
    """Where the cheap pass looks. Never open-ended - a sweep must not read a whole clip."""
    frames = max(1, sweep.frames)
    budget = max(1, sweep.clusters) * frames
    if total_frames <= 0:
        return [(0, budget)]
    # Eight clusters is the right shape for a long clip and far too many for a 5-second one,
    # where it would sweep half the frames. Dropping clusters rather than shortening them
    # keeps each one long enough for `min_cluster_hits` to mean anything.
    affordable = int(total_frames * SWEEP_MAX_COVERAGE) // frames
    clusters = max(1, min(sweep.clusters, affordable))
    return [(start, count if count is not None else budget)
            for start, count in plan_windows(total_frames, clusters, frames, radius=0)]


#: Frames of detection one ffmpeg startup is worth. Measured at ~250ms a seek against
#: ~20ms a frame, so a plan is priced in frames plus ten per window it opens.
SEEK_COST_IN_FRAMES = 10


def plan_cost(plan: Sequence[tuple[int, int | None]], total_frames: int) -> int:
    """Roughly what a sampling plan costs, in frames-of-detection."""
    frames = sum((count if count is not None else max(total_frames, 1))
                 for _, count in plan)
    return frames + SEEK_COST_IN_FRAMES * len(plan)


#: How much of the blind plan the sweep must save before it is worth running. A clip that
#: escalates pays for both passes, so a sweep that only just breaks even on paper loses
#: money on every clip that turns out to have text. The margin is roughly the share of a
#: folder expected to carry text - get it wrong and the cost is a few percent either way.
SWEEP_MARGIN = 0.8

#: Sample points below which a sweep is not worth trusting. On a very short clip the
#: coverage cap leaves room for only one or two, and deciding a clip is clean from two
#: glimpses of it is a worse trade than simply reading the thing - which is cheap anyway,
#: because it is short.
MIN_SWEEP_CLUSTERS = 3


def sweep_is_worth_it(total_frames: int, sweep: Sweep, windows: int, window_len: int,
                      radius: int) -> bool:
    """Whether the cheap pass would actually save anything on this clip.

    On a short clip the confirm pass reads the whole thing in one window anyway, so a sweep
    that opens eight ffmpeg processes to look at most of the same frames is pure overhead.
    The sweep earns its place on long clips, where blind windows sample hundreds of frames.
    """
    plan = sweep_plan(total_frames, sweep)
    if len(plan) < min(MIN_SWEEP_CLUSTERS, max(1, sweep.clusters)):
        return False
    blind = plan_cost(plan_windows(total_frames, windows, window_len, radius), total_frames)
    return plan_cost(plan, total_frames) < blind * SWEEP_MARGIN


def detector_input_size(detectors: Sequence, size: int):
    """Temporarily resize docTR's input, restoring it however the block exits.

    Reaches into the predictor's preprocessor because there is no public way to ask for it.
    If that ever moves, the sweep quietly runs at full size instead of failing - it would be
    slower, not wrong.

    A `SharedDetector` is asked rather than reached into, because with several clips in
    flight the size is a property of the asking thread, not of the model.
    """
    @contextlib.contextmanager
    def scoped():
        with contextlib.ExitStack() as stack:
            saved = []
            for detector in detectors:
                if isinstance(detector, SharedDetector):
                    stack.enter_context(detector.wants_input_size(size))
                elif size and size > 0:
                    try:
                        resize = detector.predictor.pre_processor.resize
                        saved.append((resize, resize.size))
                        resize.size = (int(size), int(size))
                    except AttributeError:
                        continue  # not a docTR predictor, or its internals moved
            try:
                yield
            finally:
                for resize, original in saved:
                    resize.size = original

    return scoped()


class SharedDetector:
    """The detectors as one thread-safe unit, so several clips can be scanned at once.

    Workers share one set of models rather than each loading their own. They are a single
    GPU either way, so concurrent forward passes would only queue on it, and docTR's
    preprocessor carries mutable state - the input size the sweep changes - that two threads
    would corrupt for each other.

    The lock is deliberately narrow: it covers the forward pass and nothing else, leaving
    every worker free to run ffmpeg, extract strokes and copy files in parallel. That is
    where the wall clock actually goes, and it is why this helps at all.

    Presents the `TextDetector` interface, so `dsf.pipeline` consumes it unchanged.
    """

    name = "shared"

    def __init__(self, detectors: Sequence):
        self._detectors = list(detectors)
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def detectors(self) -> list:
        return list(self._detectors)

    @contextlib.contextmanager
    def wants_input_size(self, size: int):
        """Record the size *this thread* wants; it is applied around the forward pass."""
        previous = getattr(self._local, "size", 0)
        self._local.size = int(size or 0)
        try:
            yield
        finally:
            self._local.size = previous

    def detect(self, frames: Sequence[np.ndarray]) -> list:
        size = getattr(self._local, "size", 0)
        with self._lock:
            with detector_input_size(self._detectors, size):
                per_detector = [d.detect(frames) for d in self._detectors]
        return [DetectorResult(detections=merge_detections(
            [res[i].detections for res in per_detector])) for i in range(len(frames))]


def sweep_clip(clip: Clip, cfg: PipelineConfig, detectors: Sequence, sweep: Sweep,
               info=None, cancel: threading.Event | None = None,
               on_progress: Callable[[int, int], None] | None = None
               ) -> tuple[list[int], int]:
    """Detection only, over a few short clusters. Returns ``(hit frames, frames seen)``."""
    from dsf.filters import GeometryFilter
    from dsf.media import probe, read_rgb

    info = info or probe(str(clip.path))
    geometry = GeometryFilter(cfg.filters, info.width, info.height)
    plan = sweep_plan(info.nb_frames, sweep)
    budget = sum(count for _, count in plan)
    size = max(1, cfg.detect.batch_size)

    def walk() -> Iterator[tuple[int, int, np.ndarray]]:
        """Every cluster in turn, as ``(cluster, absolute frame index, frame)``."""
        for cluster, (start, count) in enumerate(plan):
            stream = read_rgb(str(clip.path), seek_frame=start, max_frames=count, info=info)
            try:
                for offset, frame in enumerate(stream):
                    yield cluster, start + offset, frame
            finally:
                stream.close()

    # Prefetched across the whole plan rather than within a cluster. A cluster is only a
    # handful of frames - one batch, nothing to overlap - but opening the *next* cluster
    # means another ffmpeg startup, and at ~175ms those are most of what a sweep costs.
    # Running the reader ahead hides them behind the detection of the cluster before.
    found: dict[int, list[int]] = {}
    seen = 0
    with detector_input_size(detectors, sweep.input_size):
        source = prefetch(walk())
        try:
            for batch in _batches(source, size):
                frames = [frame for _, _, frame in batch]
                per_detector = [d.detect(frames) for d in detectors]
                for i, (cluster, index, _) in enumerate(batch):
                    dets = merge_detections([res[i].detections for res in per_detector])
                    # Geometry only. Whether a region was overlaid or filmed is the confirm
                    # pass's question; all this one decides is whether it is worth asking.
                    if geometry(dets):
                        found.setdefault(cluster, []).append(index)
                seen += len(batch)
                if on_progress is not None:
                    on_progress(seen, budget)
                if cancel is not None and cancel.is_set():
                    break
        finally:
            source.close()

    hits: list[int] = []
    for cluster in sorted(found):
        if len(found[cluster]) >= max(1, sweep.min_cluster_hits):
            hits.extend(found[cluster])
    return hits, seen


def confirm_plan(hits: Sequence[int], total_frames: int, window_len: int, radius: int,
                 max_windows: int = 4) -> list[tuple[int, int]]:
    """Full-pipeline windows centred on what the sweep found, rather than spread blindly.

    Blind windows spend most of their budget re-reading stretches the sweep already looked
    at and found nothing in. Centring on a hit also puts the evidence in the middle of the
    window, which is where the persistence gate has context either side of it to work with.
    """
    window_len = max(int(window_len), 2 * max(0, int(radius)) + 1)
    if not hits:
        return []
    groups: list[list[int]] = []
    for hit in sorted(set(int(h) for h in hits)):
        if groups and hit - groups[-1][-1] <= window_len:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    plan: list[tuple[int, int]] = []
    for group in groups[:max(1, max_windows)]:
        centre = (group[0] + group[-1]) // 2
        start = max(0, centre - window_len // 2)
        if total_frames > 0:
            start = min(start, max(0, total_frames - window_len))
        if start not in [s for s, _ in plan]:
            plan.append((start, window_len))
    return plan


class WordReader:
    """docTR's recogniser, asked one question: does this region read as words?"""

    def __init__(self, device: str = "auto", batch_size: int = 32,
                 arch: str = "crnn_vgg16_bn"):
        from doctr.models import recognition_predictor

        from dsf.detect.base import resolve_device

        self.device = resolve_device(device)
        self.predictor = recognition_predictor(arch=arch, pretrained=True,
                                               batch_size=max(1, batch_size))
        self.predictor.model = self.predictor.model.to(self.device).eval()
        # Shared by every clip worker, same as the detectors.
        self._lock = threading.Lock()

    def read(self, frame: np.ndarray, detections: Sequence) -> list[tuple[str, float]]:
        height, width = frame.shape[:2]
        crops = []
        for det in detections:
            x0, y0, x1, y1 = det.bbox
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(width, x1), min(height, y1)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crops.append(np.ascontiguousarray(frame[y0:y1, x0:x1]))
        if not crops:
            return []
        with self._lock:
            read = self.predictor(crops)
        out: list[tuple[str, float]] = []
        for item in read:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                out.append((str(item[0]), float(item[1])))
            else:
                out.append((str(item), 1.0))
        return out


def legible(words: Sequence[tuple[str, float]], reading: Reading) -> list[str]:
    """The words long enough and confident enough to be writing rather than an artefact.

    Length matters as much as confidence. A false box over a bright edge reads back as a
    single character at 0.95 - exactly as confident as a real word, and meaning nothing.
    """
    kept = []
    for text, confidence in words:
        letters = sum(1 for c in text if c.isalnum())
        if letters >= max(1, reading.min_chars) and confidence >= reading.min_confidence:
            kept.append(text)
    return kept


def read_evidence(clip: Clip, candidates: Sequence[tuple[int, list]], reader: "WordReader",
                  reading: Reading, info=None) -> tuple[list[str], list[str]]:
    """Re-read the strongest frames and ask what they say.

    Returns ``(legible words, everything read)``. Stops at the first frame that reads, and
    keeps the rest for the report - a rejected clip is far easier to argue with when you can
    see what the recogniser actually made of it.
    """
    from dsf.media import probe, read_rgb

    info = info or probe(str(clip.path))
    everything: list[str] = []
    for index, dets in list(candidates)[:max(1, reading.max_frames)]:
        if not dets:
            continue
        stream = read_rgb(str(clip.path), seek_frame=index, max_frames=1, info=info)
        try:
            frame = next(iter(stream), None)
        finally:
            stream.close()
        if frame is None:
            continue
        words = reader.read(frame, dets)
        good = legible(words, reading)
        if good:
            return good, [w for w, _ in words if w]
        everything.extend(w for w, _ in words if w)
    return [], everything


def scan_clip(clip: Clip, cfg: PipelineConfig, detectors: Sequence,
              thresholds: Thresholds, windows: int = 8, window_len: int = 45,
              exhaustive: bool = False, sweep: Sweep | None = None,
              reader: "WordReader | None" = None, reading: Reading | None = None,
              cancel: threading.Event | None = None,
              on_progress: Callable[[int, int, str], None] | None = None) -> ClipResult:
    """Decide whether one clip carries overlay text, cheapest question first.

    Three stages, each paid for only by the clips that got past the one before: is there
    anything text-shaped here at all, was it overlaid rather than filmed, and does it read
    as words.
    """
    from dsf.media import probe
    from dsf.pipeline import context_radius, iter_masks_detailed

    reading = reading or Reading()
    started = time.monotonic()
    result = ClipResult(clip=clip)
    try:
        info = probe(str(clip.path))
    except Exception as exc:  # noqa: BLE001 - one bad clip must not end the scan
        result.verdict = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        result.elapsed = time.monotonic() - started
        return result

    def progress(stage: str):
        if on_progress is None:
            return None
        return lambda seen, budget: on_progress(seen, budget, stage)

    def finish(verdict: str) -> ClipResult:
        result.verdict = verdict
        result.elapsed = time.monotonic() - started
        return result

    cancelled = lambda: cancel is not None and cancel.is_set()  # noqa: E731
    radius = context_radius(cfg)

    try:
        # ------------------------------------------------------------------ sweep
        if (sweep is not None and not exhaustive
                and not sweep_is_worth_it(info.nb_frames, sweep, windows, window_len,
                                          radius)):
            # Short clip: the confirm pass reads it whole in one window, so sweeping first
            # would open six ffmpeg processes to look at most of the same frames again.
            sweep = None
        if sweep is not None and not exhaustive:
            found, seen = sweep_clip(clip, cfg, detectors, sweep, info=info, cancel=cancel,
                                     on_progress=progress("sweep"))
            result.swept_of = int(info.nb_frames)
            result.swept_frames, result.sweep_hits = seen, len(found)
            if cancelled():
                return finish("cancelled")
            if not found:
                return finish("clean")
            plan = confirm_plan(found, info.nb_frames, window_len, radius)
        else:
            plan = plan_windows(info.nb_frames, windows, window_len, radius, exhaustive)

        # ------------------------------------------------------------------ confirm
        result.stage = "confirmed"
        budget = sum(c if c is not None else max(info.nb_frames, 1) for _, c in plan)
        verdict = Verdict(thresholds)
        pixels = float(max(1, info.width * info.height))
        tick = progress("confirm")
        # Frames whose text showed most strongly, kept with their detections so the
        # recogniser can be pointed at the best evidence rather than the first thing found.
        best: list[tuple[float, int, list]] = []

        for start, count in plan:
            if cancelled():
                break
            verdict.start_window()
            stream = iter_masks_detailed(str(clip.path), cfg, info, seek_frame=start,
                                         max_frames=count, detectors=detectors)
            try:
                for offset, (mask, dets) in enumerate(stream):
                    # Opacity-weighted area, so a credit half way through a fade counts for
                    # half of what it counts for at full strength - which is the honest
                    # reading of how much of the depth map it has wrecked.
                    coverage = float(mask.sum()) / (255.0 * pixels)
                    done = verdict.observe(start + offset, coverage, len(dets))
                    if dets:
                        best.append((coverage, start + offset, list(dets)))
                    if tick is not None:
                        tick(verdict.frames_scanned, budget)
                    if done or cancelled():
                        break
            finally:
                # Closed explicitly rather than left to the collector: the chain bottoms out
                # in an ffmpeg subprocess, and early exit is the common case here.
                stream.close()
            if verdict.flagged:
                break

        result.frames_scanned = verdict.frames_scanned
        result.text_frames = verdict.text_frames
        result.longest_run = verdict.longest_run
        result.peak_coverage = verdict.peak_coverage
        result.hits = verdict.hits

        if not verdict.flagged:
            return finish("cancelled" if cancelled() else "clean")

        # ------------------------------------------------------------------ read
        if reader is None:
            return finish("text")
        result.stage = "read"
        best.sort(key=lambda item: -item[0])
        good, everything = read_evidence(clip, [(i, d) for _, i, d in best], reader,
                                         reading, info=info)
        result.words = good or everything[:6]
        return finish("text" if good or not reading.require else "rejected")
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        return finish("error")


# --------------------------------------------------------------------------- copying

def destination_for(path: Path, source_root: Path, dest_root: Path) -> Path:
    """Mirror the source's subfolder structure, so same-named clips cannot collide."""
    try:
        rel = Path(path).resolve().relative_to(Path(source_root).resolve())
    except (ValueError, OSError):
        rel = Path(Path(path).name)
    return Path(dest_root) / rel


def depth_destination(depth_path: Path, clip_path: Path, source_root: Path,
                      dest_root: Path) -> Path:
    """Put the depth map in the same relative folder as its RGB clip, not its own.

    The two trees are often laid out differently, and a pair that arrives together should
    stay together on the way out.
    """
    rel = destination_for(clip_path, source_root, Path("."))
    return Path(dest_root) / rel.parent / Path(depth_path).name


def copy_clip(src: Path, dst: Path, move: bool = False) -> str:
    """Copy (or move) a clip. Returns what happened; never overwrites.

    A destination that already exists is left alone and reported. Overwriting would be the
    wrong default for a tool whose whole job is to gather footage: a half-written copy from
    an interrupted run looks exactly like a finished one, and only the user knows which.
    """
    src, dst = Path(src), Path(dst)
    if dst.exists():
        if dst.is_dir() or src.is_dir():
            return "exists"
        try:
            same = dst.stat().st_size == src.stat().st_size
        except OSError:
            same = False
        return "exists" if same else "exists (differs)"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dst))
        return "moved"
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)
    return "copied"


# --------------------------------------------------------------------------- the scan

@dataclass
class ScanOptions:
    rgb_root: Path
    out_root: Path
    depth_root: Path | None = None
    profile: str = "subtitles"
    recursive: bool = False
    dry_run: bool = False
    move: bool = False
    skip_existing: bool = True
    exhaustive: bool = False
    windows: int = 8
    window_len: int = 45
    detect_every: int = 0
    batch_size: int = 8
    device: str = "auto"
    use_easyocr: bool = False
    thresholds: Thresholds = field(default_factory=Thresholds)
    #: None disables the cheap first pass and scans every clip with the full pipeline.
    sweep: Sweep | None = field(default_factory=Sweep)
    reading: Reading = field(default_factory=Reading)
    workers: int = DEFAULT_WORKERS


REPORT_COLUMNS = ["clip", "verdict", "stage", "swept_frames", "sweep_hits",
                  "frames_scanned", "text_frames", "longest_run", "peak_coverage",
                  "words", "first_hits", "elapsed_s", "rgb_action", "rgb_dest",
                  "depth_source", "depth_status", "depth_action", "error"]


def run_scan(options: ScanOptions, post: Callable[[tuple], None],
             cancel: threading.Event) -> None:
    """Scan every clip under ``options.rgb_root``. Runs on the worker thread.

    Communicates only through *post*, which takes a ``(kind, *payload)`` tuple. Nothing here
    may touch a widget.
    """
    def log(message: str) -> None:
        post(("log", message))

    rgb_root = Path(options.rgb_root)
    out_root = Path(options.out_root)
    rgb_out = out_root / RGB_OUT
    depth_out = out_root / DEPTH_OUT
    depth_root = Path(options.depth_root) if options.depth_root else None

    if not rgb_root.is_dir():
        post(("fatal", f"RGB folder does not exist: {rgb_root}"))
        return
    if depth_root is not None and not depth_root.is_dir():
        post(("fatal", f"Depth folder does not exist: {depth_root}"))
        return

    clips = discover_clips(rgb_root, options.recursive, skip=(out_root, rgb_out, depth_out))
    if not clips:
        post(("fatal", f"No clips found in {rgb_root}"))
        return
    log(f"{len(clips)} clip(s) to scan under {rgb_root}")

    depth_index: dict[str, list[Path]] = {}
    if depth_root is not None:
        depth_index = index_depth(depth_root, recursive=True)
        log(f"{sum(len(v) for v in depth_index.values())} depth map(s) indexed "
            f"under {depth_root}")

    cfg = build_scan_config(options.profile, options.detect_every, options.batch_size,
                            options.device, options.use_easyocr)
    log(f"profile {options.profile}: roi={cfg.filters.roi} "
        f"persist>={cfg.filters.min_persist_frames} scroll={cfg.filters.allow_vertical_scroll} "
        f"detect_every={cfg.detect.detect_every} temporal={cfg.temporal.mode}")

    # Must happen before anything imports doctr or easyocr - both read the cache environment
    # variables at import time.
    configure_model_cache()
    try:
        from dsf.detect import build_detectors

        log(f"loading detector(s): {', '.join(cfg.detect.detectors)} "
            f"(first run downloads weights)")
        # Built once for the whole run. Per clip, model construction would cost more than
        # the detection itself on a folder of short clips - and per worker it would multiply
        # the VRAM for no gain, since the GPU serialises the passes either way.
        detectors = [SharedDetector(build_detectors(cfg.detect.detectors, cfg.detect))]
    except Exception as exc:  # noqa: BLE001
        post(("fatal", f"Could not load a text detector: {exc}\n\n"
                       f"Check the models folder and that torch is installed "
                       f"(scripts/setup.ps1)."))
        return
    log("detector ready")
    if options.sweep is not None:
        log(f"sweep: {options.sweep.clusters} clusters x {options.sweep.frames} frames at "
            f"{options.sweep.input_size or 1024}px, escalating on "
            f"{options.sweep.min_cluster_hits}+ hits in a cluster (skipped on clips short "
            f"enough that the confirm pass reads them whole)")
    else:
        log("sweep off - every clip gets the full pipeline")

    # Loaded up front rather than on the first clip that needs it, so a missing recogniser
    # is reported in the first seconds of a scan instead of an hour into one.
    reader = None
    if options.reading.require:
        try:
            log("loading the recogniser (first run downloads ~63 MB)")
            reader = WordReader(cfg.detect.device, batch_size=32)
            log(f"recogniser ready on {reader.device}: text must read as a word of "
                f"{options.reading.min_chars}+ characters at "
                f"{options.reading.min_confidence:.2f} confidence")
        except Exception as exc:  # noqa: BLE001 - degrade rather than abort
            log(f"note: recogniser unavailable ({exc}); continuing without the word check, "
                f"which means more false positives")
            reader = None

    writer = None
    handle = None
    if not options.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        try:
            handle = open(out_root / REPORT_NAME, "w", newline="", encoding="utf-8")
            writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
            writer.writeheader()
            handle.flush()
        except OSError as exc:
            log(f"note: could not open {REPORT_NAME} ({exc}); continuing without a report")
            writer, handle = None, None

    workers = max(1, int(options.workers))
    if workers > 1:
        log(f"{workers} clip workers sharing one detector - results arrive as they finish, "
            f"so the order below is not the folder's")

    def examine(clip: Clip) -> ClipResult:
        """One clip, start to verdict. Runs on a pool thread; touches nothing shared.

        Copying and the report are left to the collector, which is single-threaded, so
        nothing here needs a lock beyond the ones inside the models themselves.
        """
        if cancel.is_set():
            return ClipResult(clip=clip, verdict="cancelled", error="not started")
        if (options.skip_existing and not options.dry_run
                and destination_for(clip.path, rgb_root, rgb_out).exists()):
            return ClipResult(clip=clip, verdict="skipped",
                              error="already in output folder")

        def progress(seen: int, budget: int, stage: str) -> None:
            post(("clip", clip.label, seen, budget, stage))

        return scan_clip(clip, cfg, detectors, options.thresholds,
                         windows=options.windows, window_len=options.window_len,
                         exhaustive=options.exhaustive, sweep=options.sweep,
                         reader=reader, reading=options.reading, cancel=cancel,
                         on_progress=progress)

    totals = {"text": 0, "rejected": 0, "clean": 0, "error": 0, "skipped": 0, "cancelled": 0}
    started = time.monotonic()
    done = 0
    try:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dsf-clip")
        futures = {pool.submit(examine, clip): clip for clip in clips}
        try:
            for future in as_completed(futures):
                if cancel.is_set():
                    break
                clip = futures[future]
                done += 1
                post(("overall", done, len(clips), clip.label))
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001 - a pool thread should not be fatal
                    result = ClipResult(clip=clip, verdict="error",
                                        error=f"{type(exc).__name__}: {exc}")
                totals[result.verdict] = totals.get(result.verdict, 0) + 1

                if result.verdict == "skipped":
                    log(f"[{done}/{len(clips)}] {clip.label}: skipped, already copied")
                    post(("row", clip.label, "skipped", "already in output", "-", "-"))
                    if writer is not None:
                        _write_row(writer, handle, result)
                    continue

                rgb_dest = destination_for(clip.path, rgb_root, rgb_out)
                rgb_action, depth_action, depth_status = "", "", ""
                depth_path: Path | None = None
                if result.verdict == "error":
                    log(f"[{done}/{len(clips)}] {clip.label}: ERROR {result.error}")
                elif result.verdict == "text":
                    if depth_root is not None:
                        depth_path, depth_status = find_depth(clip, depth_index)
                    if options.dry_run:
                        rgb_action = "dry-run"
                        depth_action = "dry-run" if depth_path else ""
                    else:
                        try:
                            rgb_action = copy_clip(clip.path, rgb_dest, options.move)
                        except OSError as exc:
                            rgb_action = f"failed: {exc}"
                        if depth_path is not None:
                            dest = depth_destination(depth_path, clip.path, rgb_root,
                                                     depth_out)
                            try:
                                depth_action = copy_clip(depth_path, dest, options.move)
                            except OSError as exc:
                                depth_action = f"failed: {exc}"
                    note = f" | depth {depth_status}" if depth_root is not None else ""
                    if depth_root is not None and depth_path is None:
                        note += " (RGB copied anyway)"
                    log(f"[{done}/{len(clips)}] {clip.label}: TEXT - {result.evidence} "
                        f"-> {rgb_action or 'dry-run'}{note}")
                else:
                    log(f"[{done}/{len(clips)}] {clip.label}: {result.verdict} - "
                        f"{result.evidence}")

                post(("row", clip.label, result.verdict, result.evidence,
                      depth_status or "-", rgb_action or "-"))
                if writer is not None:
                    # Only claim a destination when something was actually put there.
                    _write_row(writer, handle, result, rgb_action,
                               str(rgb_dest) if rgb_action else "",
                               str(depth_path or ""), depth_status, depth_action)
        finally:
            if cancel.is_set():
                # Clips that never started can simply be dropped; the ones already running
                # watch the same event and stop at their next frame.
                for pending in futures:
                    pending.cancel()
                log("cancelled")
            pool.shutdown(wait=True)
    finally:
        if handle is not None:
            handle.close()

    elapsed = time.monotonic() - started
    summary = (f"{totals['text']} with text, {totals['clean']} clean, "
               f"{totals['rejected']} rejected (found text-like regions that did not read "
               f"as words), {totals['skipped']} skipped, {totals['error']} errors "
               f"in {elapsed:.1f}s")
    if options.dry_run:
        summary += " (dry run - nothing was copied)"
    elif writer is not None:
        summary += f" - report: {out_root / REPORT_NAME}"
    post(("done", summary))


def _write_row(writer, handle, result: ClipResult, rgb_action: str = "",
               rgb_dest: str = "", depth_source: str = "", depth_status: str = "",
               depth_action: str = "") -> None:
    writer.writerow({
        "clip": str(result.clip.path),
        "verdict": result.verdict,
        "stage": result.stage,
        "swept_frames": result.swept_frames,
        "sweep_hits": result.sweep_hits,
        "words": " ".join(result.words),
        "frames_scanned": result.frames_scanned,
        "text_frames": result.text_frames,
        "longest_run": result.longest_run,
        "peak_coverage": f"{result.peak_coverage:.6e}",
        "first_hits": " ".join(str(h) for h in result.hits),
        "elapsed_s": f"{result.elapsed:.2f}",
        "rgb_action": rgb_action,
        "rgb_dest": rgb_dest,
        "depth_source": depth_source,
        "depth_status": depth_status,
        "depth_action": depth_action,
        "error": result.error,
    })
    # Flushed per row so an interrupted scan still leaves a usable report.
    if handle is not None:
        handle.flush()


# --------------------------------------------------------------------------- GUI

class ScannerApp:
    """The window. Everything here runs on the main thread."""

    def __init__(self, root: "tk.Tk"):
        self.root = root
        self.queue: queue.Queue = queue.Queue()
        self.cancel = threading.Event()
        self.worker: threading.Thread | None = None
        #: clip label -> (frames seen, budget, stage), for the clips being scanned right now
        self.inflight: dict[str, tuple[int, int, str]] = {}

        root.title("dsf - find clips with subtitles or credits")
        root.geometry("1080x860")
        root.minsize(900, 680)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(4, weight=3)
        root.rowconfigure(5, weight=2)

        self.vars = {
            "rgb": tk.StringVar(),
            "depth": tk.StringVar(),
            "out": tk.StringVar(),
            "profile": tk.StringVar(value="subtitles"),
            "recursive": tk.BooleanVar(value=False),
            "dry_run": tk.BooleanVar(value=False),
            "move": tk.BooleanVar(value=False),
            "skip_existing": tk.BooleanVar(value=True),
            "exhaustive": tk.BooleanVar(value=False),
            "use_sweep": tk.BooleanVar(value=True),
            "sweep_clusters": tk.StringVar(value="8"),
            "sweep_frames": tk.StringVar(value="6"),
            "sweep_hits": tk.StringVar(value="2"),
            "require_words": tk.BooleanVar(value=True),
            "min_chars": tk.StringVar(value="3"),
            "min_confidence": tk.StringVar(value="0.45"),
            "windows": tk.StringVar(value="8"),
            "window_len": tk.StringVar(value="45"),
            "min_text_frames": tk.StringVar(value="6"),
            "min_run": tk.StringVar(value="3"),
            "min_coverage": tk.StringVar(value="2e-5"),
            "detect_every": tk.StringVar(value="0"),
            "batch_size": tk.StringVar(value="8"),
            "workers": tk.StringVar(value=str(DEFAULT_WORKERS)),
            "device": tk.StringVar(value="auto"),
            "easyocr": tk.BooleanVar(value=False),
        }
        self._load_settings()

        self._build_folders(root)
        self._build_options(root)
        self._build_actions(root)
        self._build_results(root)
        self._build_log(root)

        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._drain)

    # ----------------------------------------------------------------- layout

    def _build_folders(self, root) -> None:
        frame = ttk.LabelFrame(root, text="Folders", padding=8)
        frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        frame.columnconfigure(1, weight=1)
        rows = [
            ("RGB clips", "rgb", "Folder of RGB clips to scan"),
            ("Depth maps (optional)", "depth", "Folder of matching *_depth maps"),
            ("Output", "out", "Where rgb_with_text/ and depth_with_text/ are written"),
        ]
        for i, (label, key, title) in enumerate(rows):
            ttk.Label(frame, text=label).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=3)
            ttk.Entry(frame, textvariable=self.vars[key]).grid(row=i, column=1, sticky="ew",
                                                              pady=3)
            ttk.Button(frame, text="Browse...",
                       command=lambda k=key, t=title: self._browse(k, t)
                       ).grid(row=i, column=2, sticky="w", padx=(8, 0), pady=3)

    def _build_options(self, root) -> None:
        frame = ttk.LabelFrame(root, text="Options", padding=8)
        frame.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        for col in (1, 3, 5):
            frame.columnconfigure(col, weight=1)

        def spin(row, col, label, key, lo, hi):
            ttk.Label(frame, text=label).grid(row=row, column=col, sticky="w", padx=(0, 6),
                                              pady=3)
            ttk.Spinbox(frame, from_=lo, to=hi, width=8, textvariable=self.vars[key]
                        ).grid(row=row, column=col + 1, sticky="w", pady=3)

        ttk.Label(frame, text="Profile").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Combobox(frame, values=PROFILES, textvariable=self.vars["profile"],
                     state="readonly", width=12).grid(row=0, column=1, sticky="w", pady=3)
        ttk.Label(frame, text="Device").grid(row=0, column=2, sticky="w", padx=(12, 6))
        ttk.Combobox(frame, values=("auto", "cuda", "cpu"), textvariable=self.vars["device"],
                     state="readonly", width=8).grid(row=0, column=3, sticky="w", pady=3)
        spin(0, 4, "Batch size", "batch_size", 1, 64)
        spin(9, 2, "Clip workers", "workers", 1, 16)

        ttk.Checkbutton(frame, text="Scan subfolders", variable=self.vars["recursive"]
                        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(frame, text="Dry run (decide, copy nothing)",
                        variable=self.vars["dry_run"]
                        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(frame, text="Move instead of copy", variable=self.vars["move"]
                        ).grid(row=1, column=4, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(frame, text="Skip clips already in the output folder",
                        variable=self.vars["skip_existing"]
                        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(frame, text="Scan every frame (slow, exhaustive)",
                        variable=self.vars["exhaustive"]
                        ).grid(row=2, column=2, columnspan=2, sticky="w", pady=3)
        ttk.Checkbutton(frame, text="Add easyocr (~2x slower, better on stylised titles)",
                        variable=self.vars["easyocr"]
                        ).grid(row=2, column=4, columnspan=2, sticky="w", pady=3)

        ttk.Separator(frame, orient="horizontal").grid(row=3, column=0, columnspan=6,
                                                       sticky="ew", pady=(8, 4))
        ttk.Checkbutton(frame, text="Fast sweep first (detection only)",
                        variable=self.vars["use_sweep"]
                        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        spin(4, 2, "Sweep clusters", "sweep_clusters", 1, 64)
        spin(4, 4, "Frames each", "sweep_frames", 1, 60)

        ttk.Checkbutton(frame, text="Require the text to read as words (Latin script)",
                        variable=self.vars["require_words"]
                        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=3)
        spin(5, 2, "Min word length", "min_chars", 1, 12)
        ttk.Label(frame, text="Min word conf.").grid(row=5, column=4, sticky="w", padx=(0, 6))
        ttk.Entry(frame, textvariable=self.vars["min_confidence"], width=10
                  ).grid(row=5, column=5, sticky="w", pady=3)

        ttk.Separator(frame, orient="horizontal").grid(row=6, column=0, columnspan=6,
                                                       sticky="ew", pady=(8, 4))
        spin(7, 0, "Confirm windows", "windows", 1, 64)
        spin(7, 2, "Frames per window", "window_len", 5, 2000)
        spin(7, 4, "Detect every Nth frame", "detect_every", 0, 30)

        spin(8, 0, "Min text frames", "min_text_frames", 1, 500)
        spin(8, 2, "Min consecutive", "min_run", 1, 200)
        ttk.Label(frame, text="Min coverage").grid(row=8, column=4, sticky="w", padx=(0, 6))
        ttk.Entry(frame, textvariable=self.vars["min_coverage"], width=10
                  ).grid(row=8, column=5, sticky="w", pady=3)
        spin(9, 0, "Sweep hits to escalate", "sweep_hits", 1, 60)

        ttk.Label(frame, foreground="#666", text=(
            "The sweep runs detection alone and skips a clip outright when it finds nothing "
            "text-shaped - which is most of a folder, and where nearly all the time is "
            "saved. What survives gets the full pipeline on windows centred where the sweep "
            "looked, then the strongest frame is handed to a recogniser: a railing or a "
            "window grid can hold still and hold its colour, but it does not read as a word. "
            "Windows are consecutive runs of frames because the persistence gate - the thing "
            "that spares shop signs and licence plates - reads a detection's neighbours in "
            "time. 'Detect every Nth' of 0 keeps the profile's own value."
        ), wraplength=1000, justify="left").grid(row=10, column=0, columnspan=6, sticky="w",
                                                 pady=(6, 0))

    def _build_actions(self, root) -> None:
        frame = ttk.Frame(root, padding=(10, 4))
        frame.grid(row=2, column=0, sticky="ew")
        frame.columnconfigure(2, weight=1)
        self.start_button = ttk.Button(frame, text="Start scan", command=self.on_start)
        self.start_button.grid(row=0, column=0, sticky="w")
        self.cancel_button = ttk.Button(frame, text="Cancel", command=self.on_cancel,
                                        state="disabled")
        self.cancel_button.grid(row=0, column=1, sticky="w", padx=(8, 0))

        bars = ttk.Frame(root, padding=(10, 0))
        bars.grid(row=3, column=0, sticky="ew")
        bars.columnconfigure(1, weight=1)
        ttk.Label(bars, text="Overall").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.overall_bar = ttk.Progressbar(bars, maximum=100)
        self.overall_bar.grid(row=0, column=1, sticky="ew", pady=2)
        self.overall_label = ttk.Label(bars, text="idle", width=34, anchor="w")
        self.overall_label.grid(row=0, column=2, sticky="w", padx=(8, 0))
        ttk.Label(bars, text="Clip").grid(row=1, column=0, sticky="w", padx=(0, 8))
        self.clip_bar = ttk.Progressbar(bars, maximum=100)
        self.clip_bar.grid(row=1, column=1, sticky="ew", pady=2)
        self.clip_label = ttk.Label(bars, text="", width=34, anchor="w")
        self.clip_label.grid(row=1, column=2, sticky="w", padx=(8, 0))

    def _build_results(self, root) -> None:
        frame = ttk.LabelFrame(root, text="Results", padding=6)
        frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=4)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        columns = ("clip", "verdict", "evidence", "depth", "action")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings", height=8)
        widths = {"clip": 320, "verdict": 80, "evidence": 320, "depth": 100, "action": 120}
        for name in columns:
            self.tree.heading(name, text=name.capitalize())
            self.tree.column(name, width=widths[name], anchor="w")
        self.tree.grid(row=0, column=0, sticky="nsew")
        bar = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        bar.grid(row=0, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=bar.set)
        self.tree.tag_configure("text", foreground="#0a7d24")
        self.tree.tag_configure("error", foreground="#b00020")
        self.tree.tag_configure("skipped", foreground="#777777")
        # Passed the overlay gates but did not read as words - the clips worth eyeballing.
        self.tree.tag_configure("rejected", foreground="#a15c00")

    def _build_log(self, root) -> None:
        frame = ttk.LabelFrame(root, text="Log", padding=6)
        frame.grid(row=5, column=0, sticky="nsew", padx=10, pady=(4, 10))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        self.log_view = ScrolledText(frame, height=8, wrap="word", state="disabled")
        self.log_view.grid(row=0, column=0, sticky="nsew")

    # ----------------------------------------------------------------- actions

    def _browse(self, key: str, title: str) -> None:
        chosen = filedialog.askdirectory(title=title,
                                         initialdir=self.vars[key].get() or None)
        if chosen:
            self.vars[key].set(chosen)
            if key == "rgb" and not self.vars["out"].get():
                self.vars["out"].set(str(Path(chosen).parent / "text_triage"))

    def _show_inflight(self) -> None:
        """One bar for every clip being worked on at once.

        With several workers a per-clip bar is meaningless - it would jump between clips on
        every update - so the bar aggregates the frames in flight and the label names them.
        """
        if not self.inflight:
            self.clip_bar.configure(value=0)
            self.clip_label.configure(text="")
            return
        seen = sum(item[0] for item in self.inflight.values())
        budget = sum(item[1] for item in self.inflight.values())
        self.clip_bar.configure(maximum=max(1, budget), value=min(seen, budget))
        names = ", ".join(sorted(self.inflight))
        count = len(self.inflight)
        prefix = f"{count} in flight: " if count > 1 else ""
        self.clip_label.configure(text=f"{prefix}{names}"[:44])

    def log(self, message: str) -> None:
        self.log_view.configure(state="normal")
        self.log_view.insert("end", message + "\n")
        self.log_view.see("end")
        self.log_view.configure(state="disabled")

    def _collect(self) -> ScanOptions | None:
        """Read the widgets into a plain options object, or complain and return None."""
        rgb, out = self.vars["rgb"].get().strip(), self.vars["out"].get().strip()
        depth = self.vars["depth"].get().strip()
        if not rgb:
            messagebox.showerror("Missing folder", "Choose the folder of RGB clips.")
            return None
        if not out and not self.vars["dry_run"].get():
            messagebox.showerror("Missing folder",
                                 "Choose an output folder, or tick Dry run.")
            return None
        try:
            numbers = {
                "windows": int(self.vars["windows"].get()),
                "window_len": int(self.vars["window_len"].get()),
                "detect_every": int(self.vars["detect_every"].get()),
                "batch_size": int(self.vars["batch_size"].get()),
                "min_text_frames": int(self.vars["min_text_frames"].get()),
                "min_run": int(self.vars["min_run"].get()),
                "min_coverage": float(self.vars["min_coverage"].get()),
                "sweep_clusters": int(self.vars["sweep_clusters"].get()),
                "sweep_frames": int(self.vars["sweep_frames"].get()),
                "sweep_hits": int(self.vars["sweep_hits"].get()),
                "min_chars": int(self.vars["min_chars"].get()),
                "min_confidence": float(self.vars["min_confidence"].get()),
                "workers": int(self.vars["workers"].get()),
            }
        except ValueError as exc:
            messagebox.showerror("Bad number", f"Check the numeric fields: {exc}")
            return None

        if self.vars["move"].get() and not self.vars["dry_run"].get():
            if not messagebox.askokcancel(
                    "Move, not copy",
                    "Flagged clips will be MOVED out of the source folder.\n\n"
                    "Continue?"):
                return None

        return ScanOptions(
            rgb_root=Path(rgb),
            out_root=Path(out) if out else Path(rgb).parent / "text_triage",
            depth_root=Path(depth) if depth else None,
            profile=self.vars["profile"].get(),
            recursive=self.vars["recursive"].get(),
            dry_run=self.vars["dry_run"].get(),
            move=self.vars["move"].get(),
            skip_existing=self.vars["skip_existing"].get(),
            exhaustive=self.vars["exhaustive"].get(),
            windows=numbers["windows"],
            window_len=numbers["window_len"],
            detect_every=numbers["detect_every"],
            batch_size=numbers["batch_size"],
            device=self.vars["device"].get(),
            use_easyocr=self.vars["easyocr"].get(),
            thresholds=Thresholds(numbers["min_coverage"], numbers["min_text_frames"],
                                  numbers["min_run"]),
            sweep=Sweep(numbers["sweep_clusters"], numbers["sweep_frames"],
                        numbers["sweep_hits"]) if self.vars["use_sweep"].get() else None,
            reading=Reading(require=self.vars["require_words"].get(),
                            min_chars=numbers["min_chars"],
                            min_confidence=numbers["min_confidence"]),
            workers=numbers["workers"],
        )

    def on_start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        options = self._collect()
        if options is None:
            return
        self._save_settings()
        for row in self.tree.get_children():
            self.tree.delete(row)
        self.inflight.clear()
        self.cancel.clear()
        self.start_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.overall_bar.configure(value=0)
        self.clip_bar.configure(value=0)
        self.overall_label.configure(text="starting")
        self.log(f"--- scan started {time.strftime('%H:%M:%S')} ---")

        def work() -> None:
            try:
                run_scan(options, self.queue.put, self.cancel)
            except Exception:  # noqa: BLE001 - the worker's last line of defence
                self.queue.put(("fatal", traceback.format_exc()))
            finally:
                self.queue.put(("finished",))

        # Daemon, so a wedged ffmpeg or model load can never hold the window open.
        self.worker = threading.Thread(target=work, name="dsf-scan", daemon=True)
        self.worker.start()

    def on_cancel(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.cancel.set()
            self.cancel_button.configure(state="disabled")
            self.overall_label.configure(text="cancelling...")
            self.log("cancel requested - finishing the current frame")

    def on_close(self) -> None:
        self.cancel.set()
        self._save_settings()
        self.root.destroy()

    # ----------------------------------------------------------------- queue

    def _drain(self) -> None:
        """Pull whatever the worker has posted. The only place worker output reaches Tk."""
        try:
            while True:
                message = self.queue.get_nowait()
                self._handle(message)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _handle(self, message: tuple) -> None:
        kind = message[0]
        if kind == "log":
            self.log(message[1])
        elif kind == "overall":
            done, total, label = message[1], message[2], message[3]
            self.overall_bar.configure(maximum=max(1, total), value=done)
            self.overall_label.configure(text=f"{done}/{total}  {label}"[:44])
        elif kind == "clip":
            _, label, seen, budget, stage = message
            self.inflight[label] = (seen, budget, stage)
            self._show_inflight()
        elif kind == "row":
            _, clip, verdict, evidence, depth, action = message
            self.inflight.pop(clip, None)
            self._show_inflight()
            self.tree.insert("", "end", values=(clip, verdict, evidence, depth, action),
                             tags=(verdict,))
            self.tree.yview_moveto(1.0)
        elif kind == "fatal":
            self.log(f"ERROR: {message[1]}")
            messagebox.showerror("Scan failed", message[1])
        elif kind == "done":
            self.log(message[1])
            self.overall_label.configure(text="done")
        elif kind == "finished":
            self.start_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")
            self.inflight.clear()
            self._show_inflight()

    # ----------------------------------------------------------------- settings

    def _load_settings(self) -> None:
        try:
            saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for key, value in saved.items():
            if key in self.vars:
                try:
                    self.vars[key].set(value)
                except tk.TclError:
                    continue

    def _save_settings(self) -> None:
        try:
            SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_PATH.write_text(
                json.dumps({k: v.get() for k, v in self.vars.items()}, indent=2),
                encoding="utf-8")
        except OSError:
            pass  # a missing settings file is never worth interrupting the user over


def main() -> int:
    if tk is None:
        print("tkinter is not available in this Python.\n"
              "On Windows, reinstall Python with the 'tcl/tk and IDLE' option ticked;\n"
              "on Debian/Ubuntu, install python3-tk.", file=sys.stderr)
        return 1
    configure_model_cache()
    root = tk.Tk()
    ScannerApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
