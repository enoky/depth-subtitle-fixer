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

from . import accel
from .composite import code_range, composite_frame, resize_alpha, resolve_range
from .config import PipelineConfig
from .detect import build_detectors
from .detect.base import Detection, merge_detections
from .filters import GeometryFilter, appearance_ok, persistence_ok, sliding_window
from .media import is_sequence, open_depth_sink, probe, read_depth, read_rgb
from .prefetch import depth_for, prefetch
from .refine.strokes import AlphaPatch, extract_patch
from .temporal import from_u8, smooth
from .videoio import VideoInfo

ProgressFn = Callable[[int], None]

#: What `remembered` uses when nobody hands it a backend - which is what its own tests do,
#: and what any caller outside this module gets.
_CPU_OPS = accel.CpuOps()


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


def learned_strokes(frame: np.ndarray, cfg: PipelineConfig) -> np.ndarray | None:
    """The stroke mask a trained model reads off one frame, or None if it is not being used.

    Kept here rather than inside `extract_patch` so the model runs once for the picture
    instead of once per detection box. A forward pass costs the same either way - the input
    is resized to 1024 before it reaches the model - so N boxes would be N times the price of
    the one answer they are all cut from.

    A missing install is not an error at this level. `strokes_from` is a preference, the
    extractor falls back to the residual crop by crop, and a render that quietly used the
    slower-but-present path is better than one that stops after ninety frames.
    """
    if cfg.strokes.strokes_from != "hisam":
        return None
    from .detect.base import resolve_device
    from .refine import hisam

    if not hisam.available():
        warnings.warn("strokes_from='hisam' but the model is not installed; "
                      "run scripts/fetch_hisam.py. Falling back to the luma residual.")
        return None
    return hisam.strokes(frame, device=resolve_device(cfg.detect.device))


def depth_guide(plane: np.ndarray, info: VideoInfo, width: int, height: int,
                cfg: PipelineConfig) -> np.ndarray:
    """One depth frame as a float32 [0, 1] plane the size of the RGB frame it describes.

    Two conversions, both so that `StrokeConfig.depth_tol` means one thing everywhere.
    Normalising over the *legal* code range rather than the raw integer range makes the
    tolerance the same fraction of the picture whether the map arrived tv- or pc-range; and
    scaling to the RGB size means a crop taken from this lines up with the same crop of the
    frame, which matters because DepthCrafter routinely answers at a different resolution
    than the clip it was given. Linear rather than area: the guide is read as a level, and
    the extractor already samples the inside of each stroke to stay off the resampled edges.
    """
    lo, hi = code_range(info.bit_depth, resolve_range(cfg.composite, info.color_range))
    guide = (plane.astype(np.float32) - lo) / max(hi - lo, 1.0)
    if guide.shape != (height, width):
        guide = cv2.resize(guide, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(guide, 0.0, 1.0)


def _guides(depth_path: str, cfg: PipelineConfig, depth_info: VideoInfo, width: int,
            height: int, start: int, seek_frame: int) -> Iterator[np.ndarray]:
    for unit in read_depth(depth_path, depth_info, start=start, seek_frame=seek_frame):
        yield depth_guide(unit.plane, depth_info, width, height, cfg)


def _with_guides(frames: Iterator[np.ndarray],
                 guides: Iterator[np.ndarray] | None) -> Iterator[tuple]:
    """Pair each RGB frame with its depth guide, or with None when there is no depth map.

    A generator rather than a bare `zip` so that closing it closes both readers. `prefetch`
    owns the stream it is handed and closes it when the consumer stops early - which it does
    on every clip that has seen enough - and a zip object has no `close` to call, so the
    ffmpeg processes underneath would be left to whenever the garbage collector got round to
    them. The pair runs short as soon as either side does, which is what `check_alignment`
    warns about when the two streams are different lengths.
    """
    try:
        if guides is None:
            for frame in frames:
                yield frame, None
            return
        for pair in zip(frames, guides):
            yield pair
    finally:
        frames.close()
        if guides is not None:
            guides.close()


def iter_frame_items(rgb_path: str, cfg: PipelineConfig, info: VideoInfo,
                     start: int = 0, max_frames: int | None = None,
                     seek_frame: int = 0,
                     progress: ProgressFn | None = None,
                     detectors: Sequence | None = None,
                     depth_path: str | None = None,
                     depth_info: VideoInfo | None = None,
                     depth_start: int = 0) -> Iterator[FrameItem]:
    """Detect text and extract glyph patches, one frame at a time.

    With *depth_path* the corrupted depth map is streamed alongside the RGB and handed to the
    glyph extractor, which uses it to reject blobs that are not on the text's depth. That is
    a second decode of the same file - the compositor reads it again at render time - and it
    is left that way on purpose: holding a clip's depth to hand it back later is exactly the
    unbounded buffer this module streams to avoid, and the decode is small change next to
    detection.
    """
    detectors = list(detectors) if detectors else build_detectors(cfg.detect.detectors,
                                                                 cfg.detect)
    geometry = GeometryFilter(cfg.filters, info.width, info.height)
    every = max(1, cfg.detect.detect_every)
    chunk_size = max(1, cfg.detect.batch_size) * every

    guides = None
    if depth_path is not None:
        depth_info = depth_info or probe(depth_path)
        guides = _guides(depth_path, cfg, depth_info, info.width, info.height,
                         depth_start, seek_frame)

    # Read ahead by up to a chunk, so ffmpeg decodes the next batch while this one is going
    # through the detector instead of the two taking turns. Depth is sized against the frame
    # size rather than fixed, because everything here streams on the promise that clip
    # length is bounded by disk and not by RAM - so the budget has to know that a queued item
    # is now three bytes of RGB per pixel plus four of float32 guide, not three.
    frames = prefetch(_with_guides(read_rgb(rgb_path, start=start, seek_frame=seek_frame,
                                            info=info), guides),
                      depth=depth_for(info.width, info.height, chunk_size,
                                      channels=3 if guides is None else 7))

    try:
        yield from _detect_chunks(frames, chunk_size, every, detectors, geometry, cfg,
                                  max_frames, progress)
    finally:
        frames.close()


def _detect_chunks(frames, chunk_size: int, every: int, detectors, geometry,
                   cfg: PipelineConfig, max_frames: int | None,
                   progress: ProgressFn | None) -> Iterator[FrameItem]:
    index = 0
    carry: tuple[list[Detection], list[AlphaPatch]] = ([], [])

    for chunk in _chunks(frames, chunk_size):
        if max_frames is not None and index >= max_frames:
            break
        # Which frames in this chunk actually get run through the models.
        todo = [i for i in range(len(chunk)) if (index + i) % every == 0]
        if index == 0 and 0 not in todo:
            todo.insert(0, 0)
        batch = [chunk[i][0] for i in todo]

        per_detector = [d.detect(batch) for d in detectors]
        results: dict[int, list[Detection]] = {}
        for slot, i in enumerate(todo):
            groups = [res[slot].detections for res in per_detector]
            results[i] = merge_detections(groups)

        for i, (frame, guide) in enumerate(chunk):
            if max_frames is not None and index >= max_frames:
                return
            if i in results:
                dets = geometry(results[i])
                if cfg.filters.scene_text == "keep":
                    dets = [d for d in dets if appearance_ok(frame, d, cfg.filters)]
                # Once for the whole picture, and only when something survived the gates to
                # ask about it: a forward pass costs the same over the frame as over one
                # box, and a quarter of a second is not worth spending on a frame with no
                # text in it.
                learned = learned_strokes(frame, cfg) if dets else None
                patches = []
                for det in dets:
                    patch = extract_patch(frame, det, cfg.strokes, depth=guide,
                                          learned=learned)
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


def remembered(stream: Iterator[tuple], cfg: PipelineConfig, ops=None) -> Iterator[tuple]:
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

    *ops* is a `dsf.accel` backend; without one this runs on numpy exactly as it always did.
    The full-frame half of the stage - the thresholds, the twenty-one-frame or, the dilate -
    goes wherever that backend lives. The label bookkeeping does not: connected components
    with statistics, a bincount and a lookup have no CUDA equivalent worth having, and the
    two byte-per-pixel frames they need cost 0.27 ms each to fetch, so they are answered on
    the CPU in both backends and there is only ever one version of that arithmetic.
    """
    radius = max(0, cfg.temporal.prior_window // 2)
    if radius == 0:
        yield from stream
        return

    ops = ops or _CPU_OPS
    confident_level = cfg.temporal.prior_min_level * 255.0
    slack = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(1, cfg.temporal.prior_tolerance) + 1,) * 2)

    # How opaque a frame's text got is asked of every frame in every window it appears in -
    # once per frame here instead, because a full-frame max over a twenty-one frame window
    # cost more than the rest of the stage put together.
    tagged = ((shape, level, dets, ops.peak(level)) for shape, level, dets in stream)

    for (shape_u8, level_u8, dets, _), window in sliding_window(tagged, radius):
        found = ops.over_127(shape_u8)
        if not ops.any(found):
            yield shape_u8, level_u8, dets
            continue

        trusted = None
        for other_shape, _, _, peak in window:
            if peak < confident_level:
                continue
            # Accumulated in place: `trusted | witness` built a new full-frame array for
            # every member of the window.
            trusted = ops.or_into(trusted, ops.over_127(other_shape))
        if trusted is None or not ops.any(trusted):
            # Nothing nearby is unambiguous - no grounds to overrule this frame.
            yield shape_u8, level_u8, dets
            continue

        found = ops.download(found)
        allowed = ops.download(ops.dilate(trusted, slack)).astype(bool)

        # Judged per blob, and each blob is kept or dropped whole. A speck the fade
        # amplified has no counterpart in the remembered shape at all, while a glyph that
        # has drifted overlaps it partly - and clipping that glyph to the overlap would bite
        # pieces out of the text, which is the very artefact this is meant to remove. The
        # bar for keeping a blob is therefore *any* real support, not most of it: a faint
        # frame's mask spills a little past where the solid frames put the text.
        # Blobs are the unit of judgement, but the mask is soft - the antialiased skirt
        # around each stroke sits below the threshold and belongs to no blob at all. Each
        # of those pixels takes the verdict of the stroke it borders, because zeroing every
        # one of them would shave the edge off every glyph in the frame.
        count, labels, stats, _ = cv2.connectedComponentsWithStats(found, 8)
        areas = stats[1:count, cv2.CC_STAT_AREA]
        if not areas.size:
            yield shape_u8, level_u8, dets
            continue
        # Asked a blob at a time - `allowed[labels == label].mean()` - this reread the whole
        # frame once per glyph. Counting the labels that fall inside the remembered shape
        # answers it for every blob at once, and only over the pixels that are inside it;
        # the areas to divide by were already counted while labelling.
        inside = np.bincount(labels.ravel()[allowed.ravel()], minlength=count)[1:count]
        overlaps = inside / np.maximum(areas, 1)

        # If the memory does not describe this frame at all, the text has moved and the
        # prior stands down rather than deleting it. This is what keeps scrolling credits
        # safe: their glyphs sit where no earlier frame's glyph sat.
        weighted = float(np.average(overlaps, weights=areas))
        if weighted < cfg.temporal.prior_min_overlap:
            yield shape_u8, level_u8, dets
            continue

        # Painted through a lookup indexed by label, so the frame is written once rather
        # than once per blob kept. Written as 0/255 rather than 0/1 so that selecting with it
        # is a bitwise and once it is back on the backend - the same choice `over_127` makes.
        verdict = np.zeros(count, np.uint8)
        verdict[1:] = np.where(overlaps >= cfg.temporal.prior_min_support, 255, 0)
        if verdict[1:].all():
            yield shape_u8, level_u8, dets
            continue
        keep = ops.dilate(ops.upload(verdict[labels]), slack)
        yield ops.select(shape_u8, keep), level_u8, dets


def iter_masks_detailed(rgb_path: str, cfg: PipelineConfig, info: VideoInfo | None = None,
                        start: int = 0, max_frames: int | None = None, seek_frame: int = 0,
                        progress: ProgressFn | None = None,
                        detectors: Sequence | None = None,
                        depth_path: str | None = None,
                        depth_info: VideoInfo | None = None,
                        depth_start: int = 0
                        ) -> Iterator[tuple[np.ndarray, list[Detection]]]:
    """Yield ``(mask, accepted detections)`` per frame, fully gated and smoothed.

    The masks come back as ndarrays whichever backend produced them, so every caller of this
    is unaffected by there being a GPU. Inside, the chain stays on that backend end to end -
    a frame that went up as patches comes down once, as the finished mask.

    *depth_path* is optional throughout: without it the masks are exactly what they were.
    """
    info = info or probe(rgb_path)
    items = iter_frame_items(rgb_path, cfg, info, start=start, max_frames=max_frames,
                             seek_frame=seek_frame, progress=progress, detectors=detectors,
                             depth_path=depth_path, depth_info=depth_info,
                             depth_start=depth_start)

    # Ring depths, in frames, for the buffers the backend hands out. A buffer must outlive
    # every window that can still be holding it, so each of these is the window that holds
    # that particular result, plus slack. Sized here rather than in `dsf.accel` because these
    # are this pipeline's windows, not a property of the hardware.
    ops = accel.ops(cfg.detect.device, rings={
        # The stroke shape is let go as soon as the temporal filter has consumed it.
        "shape": cfg.temporal.window + 4,
        # The level map and the smoothed shape both ride the prior's window to the end.
        "level": cfg.temporal.prior_window + 4,
        "median": cfg.temporal.prior_window + 4,
    })

    def gated() -> Iterator[tuple]:
        radius = max(0, cfg.filters.persist_window // 2)
        for item, window in sliding_window(items, radius):
            window_dets = [w.detections for w in window]
            kept = [p for p in item.patches
                    if persistence_ok(p.det, window_dets, cfg.filters)]
            # The stroke shape and the strength it is showing at travel separately, and the
            # strength stays per-region rather than becoming one number for the frame.
            shape = ops.compose(kept, info.height, info.width, normalised=True)
            levels = ops.compose_levels(kept, info.height, info.width)
            yield ops.to_u8(shape, "shape"), ops.to_u8(levels, "level"), [p.det for p in kept]

    def smoothed() -> Iterator[tuple]:
        # Temporal filtering settles *where* the text is, never how strongly it shows.
        # Applied to the finished mask it would drag a fading credit up to its neighbours'
        # strength - so the last frame of a fade-out, with nothing detected on it at all,
        # would still get a near-solid mask stamped into depth that was never corrupted.
        radius = max(0, cfg.temporal.window // 2)
        for (shape_u8, level_u8, dets), window in sliding_window(gated(), radius):
            yield (smooth(shape_u8, [s for s, _, _ in window], cfg.temporal, ops=ops),
                   level_u8, dets)

    grow = np.ones((5, 5), np.uint8)
    for shape_u8, level_u8, dets in remembered(smoothed(), cfg, ops):
        # Levels grown a little so pixels the smoothing filled back in are covered by the
        # level of the text they belong to rather than falling off its edge.
        yield ops.download(ops.scale_by(shape_u8, ops.dilate(level_u8, grow))), dets


def iter_masks(rgb_path: str, cfg: PipelineConfig, info: VideoInfo | None = None,
               start: int = 0, max_frames: int | None = None, seek_frame: int = 0,
               progress: ProgressFn | None = None,
               detectors: Sequence | None = None,
               depth_path: str | None = None, depth_info: VideoInfo | None = None,
               depth_start: int = 0) -> Iterator[np.ndarray]:
    """Yield uint8 alpha masks at RGB resolution, fully gated and temporally smoothed."""
    for mask, _ in iter_masks_detailed(rgb_path, cfg, info, start=start,
                                       max_frames=max_frames, seek_frame=seek_frame,
                                       progress=progress, detectors=detectors,
                                       depth_path=depth_path, depth_info=depth_info,
                                       depth_start=depth_start):
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
                       progress=on_detect, depth_path=depth_path, depth_info=depth_info,
                       depth_start=depth_offset)

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
                     progress: ProgressFn | None = None,
                     depth_path: str | None = None,
                     depth_info: VideoInfo | None = None,
                     depth_start: int = 0
                     ) -> dict[int, tuple[np.ndarray, list[Detection]]]:
    """Compute ``(mask, detections)`` for a handful of specific frames.

    Each frame is processed with a small window of real context around it, so the
    persistence and temporal gates behave the same as they would in a full render - and,
    when a depth map is given, so the preview shows the same mask the render will produce.
    """
    info = info or probe(rgb_path)
    radius = context_radius(cfg)
    out: dict[int, tuple[np.ndarray, list[Detection]]] = {}
    for idx in sorted(set(int(i) for i in indices if i >= 0)):
        start = max(0, idx - radius)
        count = (idx - start) + radius + 1
        frames = list(iter_masks_detailed(rgb_path, cfg, info, seek_frame=start,
                                          max_frames=count, detectors=detectors,
                                          progress=progress, depth_path=depth_path,
                                          depth_info=depth_info, depth_start=depth_start))
        pos = idx - start
        out[idx] = frames[pos] if pos < len(frames) else \
            (np.zeros((info.height, info.width), np.uint8), [])
    return out
