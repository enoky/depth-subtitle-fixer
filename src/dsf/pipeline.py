"""Pipeline orchestration.

Everything streams. RGB frames, depth frames and masks are consumed one at a time, so clip
length is bounded by disk, not RAM. The only things buffered are a small window of glyph
patches (for the persistence gate) and a small window of uint8 masks (for temporal smoothing).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterator, Sequence

import cv2
import numpy as np

from .composite import composite_frame, resize_alpha
from .config import PipelineConfig
from .detect import build_detectors
from .detect.base import Detection, merge_detections
from .filters import GeometryFilter, appearance_ok, persistence_ok, sliding_window
from .media import is_sequence, open_depth_sink, probe, read_depth, read_rgb
from .refine.strokes import AlphaPatch, compose_alpha, compose_levels, extract_patch
from .temporal import from_u8, smooth, to_u8
from .videoio import VideoInfo

ProgressFn = Callable[[int], None]


@dataclass
class FrameItem:
    """Per-frame detection state carried through the persistence window."""

    index: int
    detections: list[Detection] = field(default_factory=list)
    patches: list[AlphaPatch] = field(default_factory=list)


def _chunks(iterable, size: int) -> Iterator[list]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_frame_items(rgb_path: str, cfg: PipelineConfig, info: VideoInfo,
                     start: int = 0, max_frames: int | None = None,
                     seek_frame: int = 0,
                     progress: ProgressFn | None = None,
                     detectors: Sequence | None = None) -> Iterator[FrameItem]:
    """Detect text and extract glyph patches, one frame at a time."""
    detectors = list(detectors) if detectors else build_detectors(cfg.detect.detectors,
                                                                 cfg.detect)
    geometry = GeometryFilter(cfg.filters, info.width, info.height)
    every = max(1, cfg.detect.detect_every)
    chunk_size = max(1, cfg.detect.batch_size) * every

    index = 0
    carry: tuple[list[Detection], list[AlphaPatch]] = ([], [])
    frames = read_rgb(rgb_path, start=start, seek_frame=seek_frame)

    for chunk in _chunks(frames, chunk_size):
        if max_frames is not None and index >= max_frames:
            break
        # Which frames in this chunk actually get run through the models.
        todo = [i for i in range(len(chunk)) if (index + i) % every == 0]
        if index == 0 and 0 not in todo:
            todo.insert(0, 0)
        batch = [chunk[i] for i in todo]

        per_detector = [d.detect(batch) for d in detectors]
        results: dict[int, list[Detection]] = {}
        for slot, i in enumerate(todo):
            groups = [res[slot].detections for res in per_detector]
            results[i] = merge_detections(groups)

        for i, frame in enumerate(chunk):
            if max_frames is not None and index >= max_frames:
                return
            if i in results:
                dets = geometry(results[i])
                if cfg.filters.scene_text == "keep":
                    dets = [d for d in dets if appearance_ok(frame, d, cfg.filters)]
                patches = []
                for det in dets:
                    patch = extract_patch(frame, det, cfg.strokes)
                    if patch is not None:
                        patches.append(patch)
                carry = (dets, patches)
            # Frames where detection was skipped inherit the previous result. This is only
            # sound for static text, which is why detect_every > 1 belongs to the
            # `subtitles` profile and the `credits` profile pins it back to 1.
            dets, patches = carry
            yield FrameItem(index=index, detections=list(dets), patches=list(patches))
            index += 1
            if progress:
                progress(index)


def remembered(stream: Iterator[tuple[np.ndarray, np.ndarray, list[Detection]]],
               cfg: PipelineConfig
               ) -> Iterator[tuple[np.ndarray, np.ndarray, list[Detection]]]:
    """Drop marks that never show up as real text anywhere nearby in time.

    Part-way through a fade the mask is normalised by whatever the text is showing at, so a
    small divisor amplifies everything else in the frame along with it - a lit edge behind
    the credit crosses the threshold and lands in the mask as a speck. Nothing inside a
    single frame separates that from a faint glyph: at 25% opacity their contrasts are
    comparable.

    Across time they are not comparable at all. The credit is on screen at full strength a
    moment later; the lit edge never is. So frames where the text is unambiguous vote on
    where text can be, and faint frames are held to that.

    The obvious hazard is scrolling credits, where a remembered shape would sit over the
    wrong place and erase the text. Hence the overlap check: the memory only filters a frame
    when it already explains most of what that frame found, so text that has moved is left
    alone rather than deleted.
    """
    radius = max(0, cfg.temporal.prior_window // 2)
    if radius == 0:
        yield from stream
        return

    confident_level = cfg.temporal.prior_min_level * 255.0
    slack = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(1, cfg.temporal.prior_tolerance) + 1,) * 2)

    for (shape_u8, level_u8, dets), window in sliding_window(stream, radius):
        found = (shape_u8 > 127).astype(np.uint8)
        if not found.any():
            yield shape_u8, level_u8, dets
            continue

        trusted = None
        for other_shape, other_level, _ in window:
            if float(other_level.max()) < confident_level:
                continue
            witness = other_shape > 127
            trusted = witness if trusted is None else (trusted | witness)
        if trusted is None or not trusted.any():
            # Nothing nearby is unambiguous - no grounds to overrule this frame.
            yield shape_u8, level_u8, dets
            continue

        allowed = cv2.dilate(trusted.astype(np.uint8), slack).astype(bool)

        # Judged per blob, and each blob is kept or dropped whole. A speck the fade
        # amplified has no counterpart in the remembered shape at all, while a glyph that
        # has drifted overlaps it partly - and clipping that glyph to the overlap would bite
        # pieces out of the text, which is the very artefact this is meant to remove. The
        # bar for keeping a blob is therefore *any* real support, not most of it: a faint
        # frame's mask spills a little past where the solid frames put the text.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(found, 8)
        # Blobs are the unit of judgement, but the mask is soft - the antialiased skirt
        # around each stroke sits below the threshold and belongs to no blob at all. Each
        # of those pixels takes the verdict of the stroke it borders, because zeroing every
        # one of them would shave the edge off every glyph in the frame.
        overlaps, areas = [], []
        for label in range(1, count):
            blob = labels == label
            overlaps.append(float(allowed[blob].mean()))
            areas.append(int(stats[label, cv2.CC_STAT_AREA]))
        if not overlaps:
            yield shape_u8, level_u8, dets
            continue

        # If the memory does not describe this frame at all, the text has moved and the
        # prior stands down rather than deleting it. This is what keeps scrolling credits
        # safe: their glyphs sit where no earlier frame's glyph sat.
        weighted = float(np.average(overlaps, weights=areas))
        if weighted < cfg.temporal.prior_min_overlap:
            yield shape_u8, level_u8, dets
            continue

        keep = np.zeros(found.shape, np.uint8)
        dropped = False
        for label, overlap in enumerate(overlaps, start=1):
            if overlap >= cfg.temporal.prior_min_support:
                keep[labels == label] = 1
            else:
                dropped = True
        if not dropped:
            yield shape_u8, level_u8, dets
            continue
        yield np.where(cv2.dilate(keep, slack).astype(bool), shape_u8, np.uint8(0)), \
            level_u8, dets


def iter_masks_detailed(rgb_path: str, cfg: PipelineConfig, info: VideoInfo | None = None,
                        start: int = 0, max_frames: int | None = None, seek_frame: int = 0,
                        progress: ProgressFn | None = None,
                        detectors: Sequence | None = None
                        ) -> Iterator[tuple[np.ndarray, list[Detection]]]:
    """Yield ``(mask, accepted detections)`` per frame, fully gated and smoothed."""
    info = info or probe(rgb_path)
    items = iter_frame_items(rgb_path, cfg, info, start=start, max_frames=max_frames,
                             seek_frame=seek_frame, progress=progress, detectors=detectors)

    def gated() -> Iterator[tuple[np.ndarray, float, list[Detection]]]:
        radius = max(0, cfg.filters.persist_window // 2)
        for item, window in sliding_window(items, radius):
            window_dets = [w.detections for w in window]
            kept = [p for p in item.patches
                    if persistence_ok(p.det, window_dets, cfg.filters)]
            # The stroke shape and the strength it is showing at travel separately, and the
            # strength stays per-region rather than becoming one number for the frame.
            shape = compose_alpha(kept, info.height, info.width, normalised=True)
            levels = compose_levels(kept, info.height, info.width)
            yield to_u8(shape), to_u8(levels), [p.det for p in kept]

    def smoothed() -> Iterator[tuple[np.ndarray, np.ndarray, list[Detection]]]:
        # Temporal filtering settles *where* the text is, never how strongly it shows.
        # Applied to the finished mask it would drag a fading credit up to its neighbours'
        # strength - so the last frame of a fade-out, with nothing detected on it at all,
        # would still get a near-solid mask stamped into depth that was never corrupted.
        radius = max(0, cfg.temporal.window // 2)
        for (shape_u8, level_u8, dets), window in sliding_window(gated(), radius):
            yield (smooth(shape_u8, [s for s, _, _ in window], cfg.temporal),
                   level_u8, dets)

    grow = np.ones((5, 5), np.uint8)
    for shape_u8, level_u8, dets in remembered(smoothed(), cfg):
        # Levels grown a little so pixels the smoothing filled back in are covered by the
        # level of the text they belong to rather than falling off its edge.
        yield to_u8(from_u8(shape_u8) * from_u8(cv2.dilate(level_u8, grow))), dets


def iter_masks(rgb_path: str, cfg: PipelineConfig, info: VideoInfo | None = None,
               start: int = 0, max_frames: int | None = None, seek_frame: int = 0,
               progress: ProgressFn | None = None,
               detectors: Sequence | None = None) -> Iterator[np.ndarray]:
    """Yield uint8 alpha masks at RGB resolution, fully gated and temporally smoothed."""
    for mask, _ in iter_masks_detailed(rgb_path, cfg, info, start=start,
                                       max_frames=max_frames, seek_frame=seek_frame,
                                       progress=progress, detectors=detectors):
        yield mask


def check_alignment(rgb: VideoInfo, depth: VideoInfo) -> list[str]:
    """Report anything about the pair that needs the user's attention."""
    notes: list[str] = []
    if (rgb.width, rgb.height) != (depth.width, depth.height):
        notes.append(
            f"resolution differs: RGB {rgb.width}x{rgb.height} vs depth "
            f"{depth.width}x{depth.height} - masks will be resampled to the depth size"
        )
    if rgb.nb_frames and depth.nb_frames and rgb.nb_frames != depth.nb_frames:
        # Only claim a real mismatch when both counts came from the container. A count
        # derived from duration x fps is a guess and would otherwise warn on every clip.
        if rgb.frames_exact and depth.frames_exact:
            notes.append(
                f"frame count differs: RGB {rgb.nb_frames} vs depth {depth.nb_frames} - "
                f"processing {min(rgb.nb_frames, depth.nb_frames)}; use "
                f"--rgb-offset/--depth-offset if they are misaligned rather than merely "
                f"different lengths"
            )
        else:
            notes.append(
                f"frame counts are estimated (RGB {rgb.nb_frames}, depth {depth.nb_frames}) "
                f"- the shorter stream ends the run"
            )
    if rgb.fps != depth.fps:
        notes.append(f"fps differs: RGB {float(rgb.fps):.4f} vs depth {float(depth.fps):.4f}")
    if depth.bit_depth < 10:
        notes.append(f"depth map is {depth.bit_depth}-bit - expected 10-bit DepthCrafter output")
    return notes


def render_from_masks(depth_path: str, masks: Iterator[np.ndarray], out_path: str,
                      cfg: PipelineConfig, depth_info: VideoInfo | None = None,
                      start: int = 0, progress: ProgressFn | None = None) -> int:
    """Composite a stream of masks onto the depth map and encode the result."""
    depth_info = depth_info or probe(depth_path)
    sink = open_depth_sink(out_path, depth_info, cfg, is_sequence(depth_path))
    written = 0
    try:
        for unit, mask_u8 in zip(read_depth(depth_path, depth_info, start=start), masks):
            alpha = resize_alpha(from_u8(mask_u8), depth_info.width, depth_info.height)
            plane = composite_frame(unit.plane, alpha, cfg.composite,
                                    depth_info.bit_depth, depth_info.color_range)
            sink.write(unit, plane)
            written += 1
            if progress:
                progress(written)
    finally:
        sink.close()
    return written


def run_fix(rgb_path: str, depth_path: str, out_path: str, cfg: PipelineConfig,
            mask_cache: str | None = None, rgb_offset: int = 0, depth_offset: int = 0,
            max_frames: int | None = None,
            on_detect: ProgressFn | None = None,
            on_render: ProgressFn | None = None) -> dict:
    """End-to-end: detect, gate, composite, encode. Optionally tees masks to a cache."""
    rgb_info, depth_info = probe(rgb_path), probe(depth_path)
    notes = check_alignment(rgb_info, depth_info)
    for note in notes:
        warnings.warn(note)

    masks = iter_masks(rgb_path, cfg, rgb_info, start=rgb_offset, max_frames=max_frames,
                       progress=on_detect)

    cache = None
    if mask_cache:
        from .maskcache import MaskCacheWriter

        cache = MaskCacheWriter(mask_cache, rgb_info.width, rgb_info.height, rgb_info.fps,
                                rgb_path, cfg.to_dict())

        def tee(source: Iterator[np.ndarray]) -> Iterator[np.ndarray]:
            for mask in source:
                cache.write(mask)
                yield mask

        masks = tee(masks)

    try:
        written = render_from_masks(depth_path, masks, out_path, cfg, depth_info,
                                    start=depth_offset, progress=on_render)
    finally:
        if cache is not None:
            cache.close()

    return {"frames": written, "notes": notes,
            "rgb": rgb_info, "depth": depth_info, "mask_cache": mask_cache}


def sample_frames(rgb_path: str, indices: Sequence[int]) -> dict[int, np.ndarray]:
    """Grab specific RGB frames by index, seeking to the first one."""
    wanted = sorted(set(int(i) for i in indices if i >= 0))
    if not wanted:
        return {}
    base, last = wanted[0], wanted[-1]
    out: dict[int, np.ndarray] = {}
    for offset, frame in enumerate(read_rgb(rgb_path, seek_frame=base)):
        idx = base + offset
        if idx in wanted:
            out[idx] = frame.copy()
        if idx >= last:
            break
    return out


def sample_depth(depth_path: str, indices: Sequence[int],
                 info: VideoInfo | None = None) -> dict[int, object]:
    """Grab specific depth frames by index, seeking to the first one."""
    wanted = sorted(set(int(i) for i in indices if i >= 0))
    if not wanted:
        return {}
    info = info or probe(depth_path)
    base, last = wanted[0], wanted[-1]
    out: dict[int, object] = {}
    for offset, frame in enumerate(read_depth(depth_path, info, seek_frame=base)):
        idx = base + offset
        if idx in wanted:
            out[idx] = frame
        if idx >= last:
            break
    return out


def context_radius(cfg: PipelineConfig) -> int:
    """How far either side of a frame the gates and the prior need to see."""
    return max(cfg.filters.persist_window // 2, cfg.temporal.window // 2,
               cfg.temporal.prior_window // 2, 1)


def context_frames(cfg: PipelineConfig, index: int) -> int:
    """How many frames must be run to produce the mask for a single *index*.

    Asking for one frame is never one frame's work - the persistence and temporal gates
    need the frames around it - so a caller showing progress needs this to mean anything.
    """
    radius = context_radius(cfg)
    start = max(0, index - radius)
    return (index - start) + radius + 1


def masks_for_frames(rgb_path: str, cfg: PipelineConfig, indices: Sequence[int],
                     info: VideoInfo | None = None,
                     detectors: Sequence | None = None,
                     progress: ProgressFn | None = None
                     ) -> dict[int, tuple[np.ndarray, list[Detection]]]:
    """Compute ``(mask, detections)`` for a handful of specific frames.

    Each frame is processed with a small window of real context around it, so the
    persistence and temporal gates behave the same as they would in a full render.
    """
    info = info or probe(rgb_path)
    radius = context_radius(cfg)
    out: dict[int, tuple[np.ndarray, list[Detection]]] = {}
    for idx in sorted(set(int(i) for i in indices if i >= 0)):
        start = max(0, idx - radius)
        count = (idx - start) + radius + 1
        frames = list(iter_masks_detailed(rgb_path, cfg, info, seek_frame=start,
                                          max_frames=count, detectors=detectors,
                                          progress=progress))
        pos = idx - start
        out[idx] = frames[pos] if pos < len(frames) else \
            (np.zeros((info.height, info.width), np.uint8), [])
    return out
