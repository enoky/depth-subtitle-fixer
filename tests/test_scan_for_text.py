"""The folder scanner: clip discovery, depth pairing, sampling, the verdict, and copying.

Everything here except the last test runs without model weights - the detection itself
belongs to `dsf` and is covered by the pipeline tests. What is new in the scanner is the
bookkeeping around it, and that is what these exercise.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import draw_subtitle, gradient_background, load_font

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scan_for_text.py"


def _load_scanner():
    """Import the script by path - it is a tool, not part of the installed package."""
    spec = importlib.util.spec_from_file_location("scan_for_text", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scan = _load_scanner()


# --------------------------------------------------------------------------- helpers

def fake_video(path: Path, size: int = 32) -> Path:
    """A file with the right extension. Discovery and pairing never open it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def fake_sequence(folder: Path, frames: int = 3) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        cv2.imwrite(str(folder / f"frame_{i:04d}.png"), np.zeros((4, 4, 3), np.uint8))
    return folder


def names(clips) -> list[str]:
    return sorted(c.path.name for c in clips)


# --------------------------------------------------------------------------- suffixes

def test_media_suffixes_come_from_the_app_and_cover_sequences():
    assert {".mp4", ".mkv", ".mov"} <= set(scan.VIDEO_SUFFIXES)
    assert ".png" in scan.MEDIA_SUFFIXES and ".png" not in scan.VIDEO_SUFFIXES


# --------------------------------------------------------------------------- discovery

def test_discovery_finds_videos_and_sequence_folders_and_ignores_the_rest(tmp_path):
    fake_video(tmp_path / "a.mp4")
    fake_video(tmp_path / "b.mkv")
    fake_video(tmp_path / "notes.txt")
    fake_sequence(tmp_path / "c_frames")
    (tmp_path / "empty_folder").mkdir()

    clips = scan.discover_clips(tmp_path)
    assert names(clips) == ["a.mp4", "b.mkv", "c_frames"]
    assert [c.sequence for c in clips if c.path.name == "c_frames"] == [True]


def test_discovery_recurses_only_when_asked(tmp_path):
    fake_video(tmp_path / "top.mp4")
    fake_video(tmp_path / "reel_02" / "deep.mp4")

    assert names(scan.discover_clips(tmp_path)) == ["top.mp4"]
    assert names(scan.discover_clips(tmp_path, recursive=True)) == ["deep.mp4", "top.mp4"]


def test_discovery_stays_out_of_its_own_output_folder(tmp_path):
    """Point input and output at the same place and the copies must not be rescanned."""
    fake_video(tmp_path / "a.mp4")
    out = tmp_path / "triage"
    fake_video(out / scan.RGB_OUT / "a.mp4")

    clips = scan.discover_clips(tmp_path, recursive=True,
                                skip=(out, out / scan.RGB_OUT, out / scan.DEPTH_OUT))
    assert names(clips) == ["a.mp4"]


def test_sequence_clip_keys_off_the_folder_name(tmp_path):
    clip = scan.Clip(fake_sequence(tmp_path / "shot_014"), sequence=True)
    assert clip.key == "shot_014"
    assert scan.Clip(tmp_path / "Shot_014.MP4").key == "shot_014"


# --------------------------------------------------------------------------- depth pairing

def test_depth_pairs_across_a_different_extension(tmp_path):
    fake_video(tmp_path / "shot_014_depth.mkv")
    index = scan.index_depth(tmp_path)
    found, status = scan.find_depth(scan.Clip(Path("shot_014.mp4")), index)
    assert status == "found" and found.name == "shot_014_depth.mkv"


def test_depth_pairing_ignores_case(tmp_path):
    fake_video(tmp_path / "SHOT_014_Depth.mp4")
    index = scan.index_depth(tmp_path)
    found, status = scan.find_depth(scan.Clip(Path("shot_014.mp4")), index)
    assert status == "found" and found is not None


def test_depth_missing_is_reported_not_raised(tmp_path):
    fake_video(tmp_path / "other_depth.mp4")
    index = scan.index_depth(tmp_path)
    found, status = scan.find_depth(scan.Clip(Path("shot_014.mp4")), index)
    assert found is None and status == "missing"


def test_a_sequence_clip_pairs_with_a_sequence_depth_folder(tmp_path):
    fake_sequence(tmp_path / "depth" / "shot_014_depth")
    index = scan.index_depth(tmp_path / "depth")
    clip = scan.Clip(Path("shot_014"), sequence=True)
    found, status = scan.find_depth(clip, index)
    assert status == "found" and found.name == "shot_014_depth"


def test_an_ambiguous_pair_prefers_the_matching_shape_and_says_so(tmp_path):
    fake_video(tmp_path / "shot_014_depth.mp4")
    fake_sequence(tmp_path / "shot_014_depth")
    index = scan.index_depth(tmp_path)

    found, status = scan.find_depth(scan.Clip(Path("shot_014"), sequence=True), index)
    assert status == "ambiguous" and found.is_dir()

    found, status = scan.find_depth(scan.Clip(Path("shot_014.mp4")), index)
    assert status == "ambiguous" and found.suffix == ".mp4"


def test_depth_index_finds_maps_in_subfolders(tmp_path):
    fake_video(tmp_path / "reel_02" / "shot_014_depth.mp4")
    index = scan.index_depth(tmp_path, recursive=True)
    found, _ = scan.find_depth(scan.Clip(Path("shot_014.mp4")), index)
    assert found is not None


# --------------------------------------------------------------------------- sampling plan

def test_a_window_is_never_shorter_than_the_gates_need():
    """Too short a window and the persistence gate loses the neighbours it judges by."""
    plan = scan.plan_windows(10_000, windows=8, window_len=5, radius=10)
    assert all(count == 21 for _, count in plan)


def test_a_short_clip_is_read_whole_rather_than_sampled():
    assert scan.plan_windows(200, windows=8, window_len=45, radius=10) == [(0, 200)]


def test_windows_spread_across_the_clip_and_reach_the_end():
    plan = scan.plan_windows(1000, windows=8, window_len=45, radius=10)
    assert len(plan) == 8
    starts = [s for s, _ in plan]
    assert starts[0] == 0
    assert starts[-1] + 45 == 1000
    assert starts == sorted(starts)


def test_an_unknown_frame_count_falls_back_to_reading_through():
    assert scan.plan_windows(0, windows=8, window_len=45, radius=10) == [(0, None)]


def test_exhaustive_mode_ignores_the_windows():
    assert scan.plan_windows(5000, windows=8, window_len=45, radius=10,
                             exhaustive=True) == [(0, None)]


# --------------------------------------------------------------------------- sweep plan

def test_the_sweep_never_asks_for_an_open_ended_read():
    """`None` means 'read to the end', which on the sweep would load a whole clip."""
    for total in (0, 40, 5000):
        assert all(count is not None and count > 0
                   for _, count in scan.sweep_plan(total, scan.Sweep()))


def test_the_sweep_spreads_its_clusters_across_a_long_clip():
    plan = scan.sweep_plan(5000, scan.Sweep(clusters=10, frames=5))
    assert len(plan) == 10
    assert plan[0][0] == 0 and plan[-1][0] + 5 == 5000
    assert sum(count for _, count in plan) == 50


def test_an_unknown_frame_count_still_bounds_the_sweep():
    assert scan.sweep_plan(0, scan.Sweep(clusters=10, frames=5)) == [(0, 50)]


def test_the_sweep_looks_in_as_many_places_as_the_confirm_pass_would():
    """Same sample points, less depth at each - so brief text is no likelier to be missed."""
    sweep = scan.Sweep()
    assert sweep.clusters == 8
    assert len(scan.sweep_plan(5000, sweep)) == \
        len(scan.plan_windows(5000, windows=8, window_len=45, radius=10))


def test_the_sweep_never_reads_most_of_a_clip():
    """Past a quarter of a clip there is nothing left for the sweep to save."""
    sweep = scan.Sweep()
    for total in (80, 150, 300, 480, 5000):
        swept = sum(count for _, count in scan.sweep_plan(total, sweep))
        assert swept <= total * scan.SWEEP_MAX_COVERAGE + sweep.frames, total


# --------------------------------------------------------------------------- cost model

def test_a_plan_is_priced_in_frames_plus_its_ffmpeg_startups():
    assert scan.plan_cost([(0, 45)], 1000) == 45 + scan.SEEK_COST_IN_FRAMES
    assert scan.plan_cost([(0, 45), (100, 45)], 1000) == 90 + 2 * scan.SEEK_COST_IN_FRAMES
    # An open-ended window means reading to the end of the clip.
    assert scan.plan_cost([(0, None)], 1000) == 1000 + scan.SEEK_COST_IN_FRAMES


def test_the_sweep_stands_down_on_clips_too_short_to_gain_from_it():
    """A very short clip is read whole; sweeping it first buys too little to trust.

    A clip that escalates pays for both passes, so the sweep has to save real work rather
    than merely break even - and on a clip too short to afford three sample points it is
    deciding 'clean' from two glimpses, which is not a trade worth making.
    """
    sweep = scan.Sweep()

    def worth(total):
        return scan.sweep_is_worth_it(total, sweep, windows=8, window_len=45, radius=10)

    assert not worth(30)
    assert not worth(60)
    assert worth(480)
    assert worth(5000)
    # Whatever the exact crossover, it must be monotonic - a longer clip never reverts to
    # being swept when a shorter one was not.
    crossings = [worth(n) for n in range(30, 3000, 30)]
    assert crossings == sorted(crossings)


# --------------------------------------------------------------------------- detector size

class FakeResize:
    def __init__(self):
        self.size = (1024, 1024)


class FakePredictor:
    def __init__(self):
        self.pre_processor = type("P", (), {"resize": FakeResize()})()


class FakeDetector:
    def __init__(self):
        self.predictor = FakePredictor()


def test_the_sweep_resizes_the_detector_and_puts_it_back():
    detector = FakeDetector()
    resize = detector.predictor.pre_processor.resize
    with scan.detector_input_size([detector], 640):
        assert resize.size == (640, 640)
    assert resize.size == (1024, 1024)


def test_the_detector_is_restored_even_when_the_sweep_raises():
    detector = FakeDetector()
    resize = detector.predictor.pre_processor.resize
    with pytest.raises(RuntimeError):
        with scan.detector_input_size([detector], 640):
            raise RuntimeError("detector blew up mid-sweep")
    assert resize.size == (1024, 1024)


def test_a_detector_without_a_preprocessor_is_left_alone():
    """EasyOCR has no docTR preprocessor; the sweep runs at full size rather than failing."""
    with scan.detector_input_size([object()], 640):
        pass
    detector = FakeDetector()
    with scan.detector_input_size([detector], 0):  # 0 means "leave it alone"
        assert detector.predictor.pre_processor.resize.size == (1024, 1024)


# --------------------------------------------------------------------------- sharing

class CountingDetector(FakeDetector):
    """Records the input size in force during each call, and how many overlap."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple] = []
        self.concurrent = 0
        self.peak_concurrent = 0
        self._lock = threading.Lock()

    def detect(self, frames):
        from dsf.detect.base import DetectorResult

        with self._lock:
            self.concurrent += 1
            self.peak_concurrent = max(self.peak_concurrent, self.concurrent)
            self.calls.append(self.predictor.pre_processor.resize.size)
        time.sleep(0.01)
        with self._lock:
            self.concurrent -= 1
        return [DetectorResult() for _ in frames]


def test_a_shared_detector_serialises_the_forward_pass():
    """One GPU, and docTR's preprocessor carries state - two passes at once corrupt it."""
    inner = CountingDetector()
    shared = scan.SharedDetector([inner])
    frames = [np.zeros((8, 8, 3), np.uint8)] * 2

    threads = [threading.Thread(target=lambda: shared.detect(frames)) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(inner.calls) == 6
    assert inner.peak_concurrent == 1, "two threads were inside the model at once"


def test_each_thread_gets_the_input_size_it_asked_for():
    """The sweep wants 640 and the confirm pass 1024, possibly on different clips at once."""
    inner = CountingDetector()
    shared = scan.SharedDetector([inner])
    frames = [np.zeros((8, 8, 3), np.uint8)]

    def ask(size):
        with scan.detector_input_size([shared], size):
            for _ in range(4):
                shared.detect(frames)

    threads = [threading.Thread(target=ask, args=(size,)) for size in (640, 0, 640, 0)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # Eight calls asked for 640 and eight for the untouched default; none saw a mixture.
    assert sorted(inner.calls) == sorted([(640, 640)] * 8 + [(1024, 1024)] * 8)


def test_the_size_is_put_back_when_a_thread_is_done():
    inner = CountingDetector()
    shared = scan.SharedDetector([inner])
    with scan.detector_input_size([shared], 512):
        shared.detect([np.zeros((8, 8, 3), np.uint8)])
    shared.detect([np.zeros((8, 8, 3), np.uint8)])
    assert inner.calls == [(512, 512), (1024, 1024)]


def test_a_shared_detector_merges_what_its_members_found():
    """It stands in for the whole detector list, so `dsf` consumes it unchanged."""
    from dsf.detect.base import Detection, DetectorResult, bbox_to_poly

    class Fixed:
        def __init__(self, box):
            self.box = box

        def detect(self, frames):
            return [DetectorResult(detections=[
                Detection(poly=bbox_to_poly(*self.box), score=0.9)]) for _ in frames]

    shared = scan.SharedDetector([Fixed((0, 0, 10, 10)), Fixed((50, 50, 60, 60))])
    results = shared.detect([np.zeros((80, 80, 3), np.uint8)] * 2)
    assert len(results) == 2
    assert all(len(r.detections) == 2 for r in results)

    # Near-duplicates from two detectors collapse, as they do inside the pipeline.
    shared = scan.SharedDetector([Fixed((0, 0, 10, 10)), Fixed((0, 0, 10, 10))])
    assert len(shared.detect([np.zeros((80, 80, 3), np.uint8)])[0].detections) == 1


# --------------------------------------------------------------------------- confirm plan

def test_the_confirm_pass_centres_its_windows_on_what_the_sweep_found():
    plan = scan.confirm_plan([1000, 1001, 1002], total_frames=5000, window_len=45, radius=10)
    assert len(plan) == 1
    start, count = plan[0]
    assert count == 45
    assert start <= 1001 <= start + count  # the evidence sits inside the window


def test_scattered_hits_become_separate_windows_up_to_a_cap():
    hits = [100, 101, 900, 901, 2000, 3000, 4000]
    plan = scan.confirm_plan(hits, total_frames=5000, window_len=45, radius=10,
                             max_windows=3)
    assert len(plan) == 3
    assert [s for s, _ in plan] == sorted(s for s, _ in plan)


def test_a_confirm_window_is_clamped_inside_the_clip():
    plan = scan.confirm_plan([4995], total_frames=5000, window_len=45, radius=10)
    start, count = plan[0]
    assert start >= 0 and start + count <= 5000
    assert scan.confirm_plan([2], total_frames=5000, window_len=45, radius=10)[0][0] == 0


def test_no_hits_means_no_confirm_work():
    assert scan.confirm_plan([], total_frames=5000, window_len=45, radius=10) == []


# --------------------------------------------------------------------------- reading

def test_a_single_confident_character_is_not_a_word():
    """The failure that motivates the length rule: a false box reads 'U' at 0.95."""
    reading = scan.Reading(min_chars=3, min_confidence=0.45)
    assert scan.legible([("U", 0.954)], reading) == []
    assert scan.legible([("never", 0.931)], reading) == ["never"]


def test_punctuation_does_not_count_towards_word_length():
    reading = scan.Reading(min_chars=3, min_confidence=0.45)
    assert scan.legible([("--.", 0.99)], reading) == []
    assert scan.legible([("this.", 0.99)], reading) == ["this."]


def test_a_confident_word_is_still_needed_not_just_a_long_one():
    reading = scan.Reading(min_chars=3, min_confidence=0.45)
    assert scan.legible([("asked", 0.20)], reading) == []


def test_one_character_repeated_is_not_a_word():
    """A fireplace grill photographs as "111" - long, confident, and meaningless."""
    reading = scan.Reading(min_chars=3, min_confidence=0.45)
    assert scan.legible([("111", 0.98)], reading) == []
    assert scan.legible([("XXX", 0.97)], reading) == []
    assert scan.legible([("0000", 0.91)], reading) == []
    # Case is not what makes a repeat, and punctuation between the repeats is not either.
    assert scan.legible([("iIiI", 0.99)], reading) == []
    assert scan.legible([("1.1.1", 0.99)], reading) == []


def test_the_repeat_rule_does_not_take_the_real_words_with_it():
    reading = scan.Reading(min_chars=3, min_confidence=0.45)
    assert scan.legible([("111", 0.99), ("HELLO", 0.88)], reading) == ["HELLO"]
    # Repeated characters within a word are ordinary; it is a word of nothing else that is not.
    assert scan.legible([("aaron", 0.9), ("111", 0.9), ("XIII", 0.9)], reading) == \
        ["aaron", "XIII"]


def test_a_repeat_short_enough_to_be_a_word_is_still_judged_on_length():
    """The two rules are independent: neither is a way round the other."""
    reading = scan.Reading(min_chars=1, min_confidence=0.45)
    assert scan.legible([("11", 0.99)], reading) == []
    assert scan.legible([("1", 0.99)], reading) == ["1"]


# --------------------------------------------------------------------------- sitting level

def ink(mask: np.ndarray):
    """A detection around everything marked in *mask*, as the pipeline would hand one over."""
    from dsf.detect.base import Detection, bbox_to_poly

    ys, xs = np.nonzero(mask > 8)
    return Detection(poly=bbox_to_poly(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def written(text: str, angle: float = 0.0, font_size: int = 36) -> np.ndarray:
    """The mask of a line of text, turned by *angle* degrees anticlockwise."""
    from conftest import draw_subtitle_full

    _, _, full = draw_subtitle_full(gradient_background(640, 360), text,
                                    font_size=font_size, y_frac=0.5)
    if angle:
        h, w = full.shape
        turn = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        full = cv2.warpAffine(full, turn, (w, h), flags=cv2.INTER_LINEAR)
    return full


def tilt_of(mask: np.ndarray, level: "scan.Level | None" = None):
    return scan.baseline_tilt(mask, ink(mask), level or scan.Level())


def test_level_writing_reads_as_level():
    for angle in (0, -3, 3, 6):
        mask = written("HELLO WORLD", angle)
        assert abs(tilt_of(mask)) <= scan.Level().max_tilt, angle


def test_writing_that_slopes_is_measured_and_thrown_out():
    for angle in (-10, 10, -20, 20, -35, 35):
        tilt = tilt_of(written("HELLO WORLD", angle))
        assert abs(tilt) > scan.Level().max_tilt, f"{angle} read as {tilt}"
        # Not just "sloped" - the angle itself comes back, which is what makes a rejection
        # arguable when a clip turns out to have been flagged wrongly.
        assert abs(abs(tilt) - abs(angle)) <= 4, f"{angle} read as {tilt}"


def test_a_sloping_grill_is_sloped_and_a_level_one_is_not():
    """The false positive this is for: structure that survives every gate upstream."""
    sloping = np.zeros((360, 640), np.uint8)
    level = np.zeros((360, 640), np.uint8)
    for bar in range(6):
        for x in range(200, 440):
            sloping[150 + bar * 10 + int((x - 200) * 0.36), x] = 255
        level[150 + bar * 10:153 + bar * 10, 200:440] = 255

    assert abs(tilt_of(sloping)) > scan.Level().max_tilt
    # A level grill is level, and is left to the gates that can tell a grill from writing.
    assert tilt_of(level) == 0.0


def test_ink_with_no_direction_is_called_level_rather_than_guessed_at():
    """A blob's best angle wins by nothing, and a winner by nothing is not believed."""
    blob = np.zeros((360, 640), np.uint8)
    blob[160:200, 200:440] = 255
    assert tilt_of(blob) == 0.0

    speckle = np.zeros((360, 640), np.uint8)
    speckle[150:210, 200:440] = (np.random.default_rng(0).random((60, 240)) > 0.6) * 255
    assert tilt_of(speckle) == 0.0


def test_a_region_too_small_or_too_narrow_to_have_a_baseline_is_not_judged():
    lone_glyph = written("I", 0, font_size=40)
    assert tilt_of(lone_glyph) is None

    pole = np.zeros((360, 640), np.uint8)
    pole[80:300, 300:312] = 255
    assert tilt_of(pole) is None

    speck = np.zeros((360, 640), np.uint8)
    speck[100:104, 200:230] = 255
    assert tilt_of(speck) is None


def test_a_short_word_is_let_through_rather_than_guessed_at():
    """Three glyphs climbing is not evidence of a slope, and must not cost a real subtitle."""
    for word in ("The", "gap", "you", "why"):
        assert abs(tilt_of(written(word))) <= scan.Level().max_tilt, word


def test_italics_lean_without_sloping():
    """The glyphs slant; the baseline does not, and it is the baseline being asked about."""
    for word in ("HELLO WORLD", "you know", "The"):
        for lean in (0.20, 0.36, -0.30):  # 0.36 is a steeper italic than any subtitle font
            mask = written(word)
            rows, cols = mask.shape
            skew = np.float32([[1, lean, -lean * rows / 2], [0, 1, 0]])
            mask = cv2.warpAffine(mask, skew, (cols, rows), flags=cv2.INTER_LINEAR)
            assert abs(tilt_of(mask)) <= scan.Level().max_tilt, f"{word} at {lean}"


def test_a_drop_shadow_is_not_a_slope():
    text = written("HELLO WORLD")
    shadowed = np.maximum(text, np.roll(np.roll(text, 3, axis=0), 3, axis=1) // 2)
    assert tilt_of(shadowed) == 0.0


def test_the_tolerance_is_what_decides_the_verdict():
    lax = scan.Level(max_tilt=30.0)
    strict = scan.Level(max_tilt=2.0)
    mask = written("HELLO WORLD", 12)
    assert abs(tilt_of(mask, lax)) <= lax.max_tilt
    assert abs(tilt_of(mask, strict)) > strict.max_tilt


def test_a_frame_keeps_its_level_regions_and_drops_the_sloping_ones():
    level, sloped = written("HELLO WORLD"), np.roll(written("MOTEL SIGN", 22), -120, axis=0)
    frame = np.maximum(level, sloped)
    kept, tilted = scan.upright_detections(frame, [ink(level), ink(sloped)], scan.Level())
    assert [d.bbox for d in kept] == [ink(level).bbox] and tilted == 1


def test_a_frame_with_nothing_but_sloping_regions_has_no_text_on_it():
    sloped = written("MOTEL SIGN", 22)
    kept, tilted = scan.upright_detections(sloped, [ink(sloped)], scan.Level())
    assert kept == [] and tilted == 1


def test_regions_too_small_to_judge_do_not_carry_a_frame_on_their_own():
    """Unjudgeable regions ride along with a level one, but cannot stand in for one."""
    sloped, glyph = written("MOTEL SIGN", 22), np.roll(written("I", 0, 40), -120, axis=0)
    frame = np.maximum(sloped, glyph)
    dets = [ink(sloped), ink(glyph)]
    assert scan.upright_detections(frame, dets, scan.Level())[0] == []

    level = np.roll(written("HELLO WORLD"), -60, axis=0)
    frame = np.maximum(frame, level)
    kept, _ = scan.upright_detections(frame, dets + [ink(level)], scan.Level())
    assert len(kept) == 2  # the lone glyph and the subtitle, not the sign


# --------------------------------------------------------------------------- the verdict

def feed(verdict, pattern, coverage=1e-3, detections=1):
    for i, hit in enumerate(pattern):
        verdict.observe(i, coverage if hit else 0.0, detections if hit else 0)
    return verdict


def test_a_clip_needs_both_enough_frames_and_a_run():
    t = scan.Thresholds(min_coverage=1e-5, min_text_frames=6, min_run=3)
    # Six text frames, but never two in a row - a detector twitching, not a subtitle.
    scattered = feed(scan.Verdict(t), [1, 0] * 6)
    assert scattered.text_frames == 6 and not scattered.flagged

    # A short run, but nothing like enough of them.
    brief = feed(scan.Verdict(t), [1, 1, 1] + [0] * 10)
    assert brief.longest_run == 3 and not brief.flagged

    assert feed(scan.Verdict(t), [1] * 6).flagged


def test_a_speck_below_the_coverage_floor_is_not_a_subtitle():
    t = scan.Thresholds(min_coverage=1e-4, min_text_frames=2, min_run=2)
    assert not feed(scan.Verdict(t), [1] * 10, coverage=1e-6).flagged
    assert feed(scan.Verdict(t), [1] * 10, coverage=1e-3).flagged


def test_coverage_without_an_accepted_detection_does_not_count():
    t = scan.Thresholds(min_coverage=1e-5, min_text_frames=2, min_run=2)
    verdict = scan.Verdict(t)
    for i in range(10):
        verdict.observe(i, 1e-3, detections=0)
    assert verdict.text_frames == 0 and not verdict.flagged


def test_a_run_does_not_carry_across_a_window_boundary():
    """Frames either side of a seek are not neighbours, so a run cannot span them."""
    t = scan.Thresholds(min_coverage=1e-5, min_text_frames=4, min_run=3)
    verdict = scan.Verdict(t)
    feed(verdict, [1, 1])
    verdict.start_window()
    feed(verdict, [1, 1])
    assert verdict.text_frames == 4 and verdict.longest_run == 2 and not verdict.flagged


def test_the_verdict_reports_where_it_saw_text():
    t = scan.Thresholds(min_coverage=1e-5, min_text_frames=2, min_run=2)
    verdict = feed(scan.Verdict(t), [0, 1, 1, 0])
    assert verdict.hits == [1, 2]
    assert verdict.peak_coverage == pytest.approx(1e-3)


# --------------------------------------------------------------------------- destinations

def test_destinations_mirror_the_source_tree(tmp_path):
    root, out = tmp_path / "rgb", tmp_path / "out" / scan.RGB_OUT
    clip = root / "reel_02" / "shot.mp4"
    fake_video(clip)
    assert scan.destination_for(clip, root, out) == out / "reel_02" / "shot.mp4"


def test_a_depth_map_lands_beside_its_clip_not_in_its_own_layout(tmp_path):
    root = tmp_path / "rgb"
    clip = fake_video(root / "reel_02" / "shot.mp4")
    depth = fake_video(tmp_path / "depth" / "anywhere" / "shot_depth.mkv")
    out = tmp_path / "out" / scan.DEPTH_OUT
    assert scan.depth_destination(depth, clip, root, out) == \
        out / "reel_02" / "shot_depth.mkv"


def test_a_clip_from_outside_the_root_keeps_just_its_name(tmp_path):
    stray = fake_video(tmp_path / "elsewhere" / "shot.mp4")
    out = tmp_path / "out"
    assert scan.destination_for(stray, tmp_path / "rgb", out) == out / "shot.mp4"


# --------------------------------------------------------------------------- copying

def test_copying_a_file_and_a_sequence(tmp_path):
    src = fake_video(tmp_path / "in" / "a.mp4", size=64)
    dst = tmp_path / "out" / "a.mp4"
    assert scan.copy_clip(src, dst) == "copied"
    assert dst.read_bytes() == src.read_bytes()
    assert dst.stat().st_mtime == pytest.approx(src.stat().st_mtime, abs=2)

    folder = fake_sequence(tmp_path / "in" / "frames", frames=2)
    out = tmp_path / "out" / "frames"
    assert scan.copy_clip(folder, out) == "copied"
    assert len(list(out.glob("*.png"))) == 2


def test_an_existing_destination_is_never_overwritten(tmp_path):
    src = fake_video(tmp_path / "in" / "a.mp4", size=64)
    dst = fake_video(tmp_path / "out" / "a.mp4", size=8)

    assert scan.copy_clip(src, dst) == "exists (differs)"
    assert dst.stat().st_size == 8

    same = fake_video(tmp_path / "out" / "b.mp4", size=64)
    assert scan.copy_clip(src, same) == "exists"


def test_moving_takes_the_clip_out_of_the_source_folder(tmp_path):
    src = fake_video(tmp_path / "in" / "a.mp4")
    dst = tmp_path / "out" / "a.mp4"
    assert scan.copy_clip(src, dst, move=True) == "moved"
    assert dst.exists() and not src.exists()


# --------------------------------------------------------------------------- config

def test_the_scene_text_gates_are_always_on():
    """`keep` is what activates the ROI, appearance and persistence gates.

    Every other knob is the user's; this one is not, because turning it off would mask shop
    signs and licence plates and the scanner would flag every clip with a car in it.
    """
    for profile in scan.PROFILES:
        assert scan.build_scan_config(profile).filters.scene_text == "keep"


def test_the_profile_sets_the_gates_and_explicit_knobs_win():
    subs = scan.build_scan_config("subtitles")
    credits = scan.build_scan_config("credits")
    assert subs.filters.roi == "bottom:0.30" and not subs.filters.allow_vertical_scroll
    assert credits.filters.roi == "full" and credits.filters.allow_vertical_scroll

    assert scan.build_scan_config("subtitles").detect.detect_every == 2
    assert scan.build_scan_config("subtitles", detect_every=5).detect.detect_every == 5
    # 0 means "whatever the profile says", so the box can be left alone.
    assert scan.build_scan_config("credits", detect_every=0).detect.detect_every == 1


# --------------------------------------------------------------------------- text size

def test_the_scan_asks_for_taller_text_than_the_fixer_does():
    """Raised for the scan alone. `dsf fix` still paints over text this would ignore."""
    from dsf.config import PipelineConfig

    base = scan.build_scan_config("subtitles")
    raised = scan.with_size_floor(base, scan.Size(), frame_height=1080)
    assert raised.filters.min_text_height > base.filters.min_text_height
    assert PipelineConfig().filters.min_text_height == base.filters.min_text_height


def test_the_floor_is_the_taller_of_the_fraction_and_the_pixels():
    size = scan.Size(min_height=0.02, min_pixels=14)
    cfg = scan.build_scan_config("subtitles")

    # 1080p: 2% is 21.6px, well past the pixel floor, so the fraction decides.
    assert scan.with_size_floor(cfg, size, 1080).filters.min_text_height == \
        pytest.approx(0.02)
    # 360p: 2% is 7.2px, which reads nothing, so the pixel floor takes over.
    assert scan.with_size_floor(cfg, size, 360).filters.min_text_height == \
        pytest.approx(14 / 360)


def test_asking_for_nothing_leaves_the_pipeline_alone():
    """0 is 'keep the profile's own value', as it is for detect_every."""
    cfg = scan.build_scan_config("subtitles")
    assert scan.with_size_floor(cfg, scan.Size(0.0, 0), 1080) is cfg
    # And a floor below the pipeline's own is not a way to *lower* it.
    assert scan.with_size_floor(cfg, scan.Size(0.001, 1), 1080) is cfg


def test_the_floor_reaches_the_gate_that_enforces_it():
    """The knob is only worth having if GeometryFilter actually reads it."""
    from dsf.detect.base import Detection, bbox_to_poly
    from dsf.filters import GeometryFilter

    cfg = scan.with_size_floor(scan.build_scan_config("credits"), scan.Size(), 1080)
    gate = GeometryFilter(cfg.filters, 1920, 1080)
    subtitle = Detection(poly=bbox_to_poly(600, 900, 1320, 950))   # 50px, ~4.6%
    speck = Detection(poly=bbox_to_poly(600, 900, 700, 916))       # 16px, ~1.5%
    assert gate.keep(subtitle) and not gate.keep(speck)
    # That speck is exactly what the unraised pipeline would have kept.
    assert GeometryFilter(scan.build_scan_config("credits").filters, 1920, 1080).keep(speck)


def test_easyocr_is_unioned_in_only_when_asked():
    assert scan.build_scan_config("both").detect.detectors == ("doctr",)
    assert scan.build_scan_config("both", use_easyocr=True).detect.detectors == \
        ("doctr", "easyocr")


# --------------------------------------------------------------------------- the window

@pytest.mark.skipif(scan.tk is None, reason="this Python has no tkinter")
def test_the_window_builds_and_the_queue_reaches_it(tmp_path, monkeypatch):
    """Build every widget for real, then push worker messages the way the scan does.

    The worker may never touch a widget, so the queue is the only route in - and a typo in
    a message shape would otherwise only surface half way through a real scan.
    """
    monkeypatch.setattr(scan, "SETTINGS_PATH", tmp_path / "settings.json")
    try:
        root = scan.tk.Tk()
    except scan.tk.TclError as exc:  # a session with no display
        pytest.skip(f"no display available: {exc}")
    try:
        app = scan.ScannerApp(root)
        root.update_idletasks()
        assert str(app.start_button["state"]) == "normal"

        for message in (("log", "hello"),
                        ("overall", 1, 3, "a.mp4"),
                        ("clip", "a.mp4", 12, 45, "sweep"),
                        ("clip", "b.mp4", 8, 45, "confirm")):
            app.queue.put(message)
        app._drain()
        root.update_idletasks()

        # Two clips in flight: the bar aggregates them, because a per-clip bar would jump
        # between workers on every update.
        assert app.overall_bar["value"] == 1
        assert app.clip_bar["value"] == 20 and app.clip_bar["maximum"] == 90
        assert "2 in flight" in app.clip_label["text"]

        for message in (("row", "a.mp4", "text", "6/6 frames", "found", "copied"),
                        ("done", "1 with text"),
                        ("finished",)):
            app.queue.put(message)
        app._drain()
        root.update_idletasks()

        assert len(app.tree.get_children()) == 1
        assert "hello" in app.log_view.get("1.0", "end")
        # `finished` clears whatever was still in flight.
        assert app.clip_bar["value"] == 0 and app.inflight == {}

        # Every box is a knob on something, and a knob wired to nothing is worse than no
        # knob. Checked in this test rather than its own because a second Tk root cannot be
        # built after the first has been destroyed.
        app.vars["rgb"].set(str(tmp_path))
        app.vars["out"].set(str(tmp_path / "out"))
        app.vars["max_tilt"].set("12.5")
        app.vars["min_text_height"].set("3.5")
        options = app._collect()
        assert options.level == scan.Level(require=True, max_tilt=12.5)
        # The box is a percentage; everything past it is a fraction.
        assert options.size.min_height == pytest.approx(0.035)

        app.vars["require_level"].set(False)
        app.vars["require_words"].set(False)
        relaxed = app._collect()
        assert not relaxed.level.require and not relaxed.reading.require

        app._save_settings()
        assert "profile" in (tmp_path / "settings.json").read_text(encoding="utf-8")
    finally:
        root.destroy()


# --------------------------------------------------------------------------- end to end

def scene_sign(background: np.ndarray, text: str, centre, font_size: int = 30,
               tint=(150, 120, 90), alpha: float = 0.55) -> np.ndarray:
    """Text as the camera would have found it: skewed, soft, and low contrast.

    Not the same thing as burned-in text drawn flat onto the finished frame. A shop sign is
    photographed - it sits at an angle, the lens softens it, and it takes the scene's own
    light - and every one of those is what the appearance gate reads.
    """
    from PIL import Image, ImageDraw

    h, w = background.shape[:2]
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text((centre[0], centre[1]), text, font=load_font(font_size),
                               fill=255)
    mask = np.array(layer).astype(np.float32) / 255.0

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[0, 0], [w, 12], [w, h - 18], [0, h]])  # a plane seen off-axis
    mask = cv2.warpPerspective(mask, cv2.getPerspectiveTransform(src, dst), (w, h))
    mask = cv2.GaussianBlur(mask, (0, 0), 1.8) * alpha

    paint = np.array(tint, dtype=np.float32)[None, None, :]
    out = background.astype(np.float32) * (1 - mask[..., None]) + paint * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def write_sequence(folder: Path, frames) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        cv2.imwrite(str(folder / f"frame_{i:04d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return folder


@pytest.mark.slow
def test_subtitles_are_flagged_and_scene_text_is_not(tmp_path):
    """The whole point of the tool, against real detections.

    Two clips over the same moving background: one with a burned-in subtitle, one with
    nothing but signage the camera photographed. Only the first may be flagged.
    """
    width, height, frames = 640, 360, 24
    base = gradient_background(width, height)

    subtitled, signs = [], []
    for i in range(frames):
        moving = np.roll(base, i * 3, axis=1)
        subtitled.append(draw_subtitle(moving, "HELLO WORLD", font_size=34,
                                       y_frac=0.85)[0])
        # One sign high in the frame, one down in the subtitle band - the second is the
        # test that matters, since the ROI alone would spare the first.
        lit = scene_sign(moving, "MOTEL", (60 + i, 70), font_size=30)
        signs.append(scene_sign(lit, "AB 51 DQ", (330 + i, 300), font_size=22,
                                tint=(120, 125, 130), alpha=0.5))

    rgb = tmp_path / "rgb"
    write_sequence(rgb / "talky", subtitled)
    write_sequence(rgb / "street", signs)

    scan.configure_model_cache()
    from dsf.detect import build_detectors

    cfg = scan.build_scan_config("subtitles")
    detectors = build_detectors(cfg.detect.detectors, cfg.detect)

    clips = {c.path.name: c for c in scan.discover_clips(rgb)}
    assert set(clips) == {"talky", "street"}

    talky = scan.scan_clip(clips["talky"], cfg, detectors, scan.Thresholds())
    street = scan.scan_clip(clips["street"], cfg, detectors, scan.Thresholds())

    assert talky.verdict == "text", talky.evidence
    assert talky.longest_run >= 3
    assert street.verdict == "clean", street.evidence


@pytest.mark.slow
def test_the_cheap_stages_reach_the_same_verdict_as_the_full_pipeline(tmp_path):
    """The sweep and the word check must change the cost, not the answer.

    A pre-filter that is faster and *wrong* is worth nothing here, so the staged scan is
    checked against the same clips run the long way round.
    """
    width, height, frames = 640, 360, 24
    base = gradient_background(width, height)

    subtitled, signs = [], []
    for i in range(frames):
        moving = np.roll(base, i * 3, axis=1)
        subtitled.append(draw_subtitle(moving, "HELLO WORLD", font_size=34,
                                       y_frac=0.85)[0])
        lit = scene_sign(moving, "MOTEL", (60 + i, 70), font_size=30)
        signs.append(scene_sign(lit, "AB 51 DQ", (330 + i, 300), font_size=22,
                                tint=(120, 125, 130), alpha=0.5))

    rgb = tmp_path / "rgb"
    write_sequence(rgb / "talky", subtitled)
    write_sequence(rgb / "street", signs)

    scan.configure_model_cache()
    from dsf.detect import build_detectors

    cfg = scan.build_scan_config("subtitles")
    detectors = build_detectors(cfg.detect.detectors, cfg.detect)
    reader = scan.WordReader(cfg.detect.device)
    clips = {c.path.name: c for c in scan.discover_clips(rgb)}

    for name, expected in (("talky", "text"), ("street", "clean")):
        staged = scan.scan_clip(clips[name], cfg, detectors, scan.Thresholds(),
                                sweep=scan.Sweep(), reader=reader, reading=scan.Reading())
        plain = scan.scan_clip(clips[name], cfg, detectors, scan.Thresholds())
        assert staged.verdict == expected, f"{name}: {staged.stage} {staged.evidence}"
        assert plain.verdict == expected, f"{name}: {plain.evidence}"

    # And the words are recorded, which is what makes a verdict arguable after the fact.
    flagged = scan.scan_clip(clips["talky"], cfg, detectors, scan.Thresholds(),
                             sweep=scan.Sweep(), reader=reader, reading=scan.Reading())
    assert flagged.stage == "read"
    assert any("HELLO" in w.upper() or "WORLD" in w.upper() for w in flagged.words), \
        flagged.words
