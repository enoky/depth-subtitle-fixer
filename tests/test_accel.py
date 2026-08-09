"""The GPU backend against the numpy original it accelerates.

`dsf.accel` exists only to make the mask chain faster, so the thing worth testing is that it
changed nothing. Every operation is held against `CpuOps` - which is the numpy code the
pipeline used before, not a second copy of it - on frames the size the pipeline really works
on, because several of these operations behave differently at 8x8 than at 1080p.

Skipped wholesale without a CUDA-enabled OpenCV, which is the ordinary state of a machine that
has not run `scripts/install_opencv_cuda.ps1`.
"""

from __future__ import annotations

import numpy as np
import pytest

import cv2
from dsf.accel import CpuOps, cuda_available, ops, resolve
from dsf.detect.base import Detection, bbox_to_poly
from dsf.refine.strokes import AlphaPatch

pytestmark = pytest.mark.skipif(not cuda_available(),
                                reason="no CUDA-enabled OpenCV; run scripts/install_opencv_cuda.ps1")

#: A real frame size. Several of these stages are only interesting at scale - the ring
#: buffers, the morphology border handling - and a toy array exercises none of it.
H, W = 540, 960


@pytest.fixture
def cpu():
    return CpuOps()


@pytest.fixture
def gpu():
    from dsf.accel import CudaOps

    backend = CudaOps()
    backend.reserve({"shape": 8, "level": 26, "median": 26})
    return backend


@pytest.fixture
def rng():
    return np.random.default_rng(20260809)


def u8(rng, shape=(H, W)) -> np.ndarray:
    return rng.integers(0, 256, shape).astype(np.uint8)


def patches(rng, count=3, height=H, width=W) -> list[AlphaPatch]:
    """Patches shaped like real detections: wide, short, scattered, part-transparent."""
    out = []
    for _ in range(count):
        ph, pw = int(rng.integers(30, 80)), int(rng.integers(200, 400))
        out.append(AlphaPatch(
            x0=int(rng.integers(0, width - pw)), y0=int(rng.integers(0, height - ph)),
            alpha=(rng.random((ph, pw), np.float32) ** 3),
            det=Detection(poly=bbox_to_poly(0, 0, pw, ph)),
            level=float(rng.uniform(0.3, 1.0)),
        ))
    return out


# --------------------------------------------------------------------------- selection

def test_resolve_honours_an_explicit_cpu(monkeypatch):
    monkeypatch.delenv("DSF_ACCEL", raising=False)
    assert resolve("cpu") == "cpu"


def test_the_environment_override_wins(monkeypatch):
    monkeypatch.setenv("DSF_ACCEL", "cpu")
    assert resolve("cuda") == "cpu"


def test_ops_are_per_thread():
    """Two workers must never share a stream, a filter or a scratch buffer."""
    import threading

    seen = []
    barrier = threading.Barrier(2)

    def grab():
        barrier.wait()  # both inside at once, so neither can simply reuse the other's
        seen.append(ops("cuda"))

    threads = [threading.Thread(target=grab) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(seen) == 2
    assert seen[0] is not seen[1]


# --------------------------------------------------------------------------- composition

def test_compose_matches_numpy(cpu, gpu, rng):
    ps = patches(rng)
    for normalised in (True, False):
        want = cpu.compose(ps, H, W, normalised)
        got = gpu.download(gpu.compose(ps, H, W, normalised))
        np.testing.assert_array_equal(got, want)


def test_compose_levels_matches_numpy(cpu, gpu, rng):
    ps = patches(rng)
    np.testing.assert_array_equal(gpu.download(gpu.compose_levels(ps, H, W)),
                                  cpu.compose_levels(ps, H, W))


def test_a_patch_hanging_off_the_edge_is_clipped_the_same_way(cpu, gpu, rng):
    """`extract_patch` clamps to the left and top but not the right and bottom."""
    edge = [AlphaPatch(x0=W - 50, y0=H - 20, alpha=rng.random((60, 120), np.float32),
                       det=Detection(poly=bbox_to_poly(0, 0, 120, 60)), level=0.7)]
    np.testing.assert_array_equal(gpu.download(gpu.compose(edge, H, W, False)),
                                  cpu.compose(edge, H, W, False))


def test_no_patches_gives_an_empty_frame(cpu, gpu):
    np.testing.assert_array_equal(gpu.download(gpu.compose([], H, W, True)),
                                  cpu.compose([], H, W, True))


# --------------------------------------------------------------------------- conversion

def test_to_u8_is_bit_exact(cpu, gpu, rng):
    alpha = rng.random((H, W), np.float32)
    np.testing.assert_array_equal(gpu.download(gpu.to_u8(gpu.upload(alpha))), cpu.to_u8(alpha))


def test_to_u8_agrees_on_exact_halves(cpu, gpu):
    """The one place the two rounding modes could differ, so it is checked directly.

    `np.rint` rounds halves to even and OpenCV's `cvRound` does too, which is why this holds -
    it is not a coincidence worth leaving untested.
    """
    halves = ((np.arange(H * W) % 511) / 2.0 / 255.0).reshape(H, W).astype(np.float32)
    np.testing.assert_array_equal(gpu.download(gpu.to_u8(gpu.upload(halves))),
                                  cpu.to_u8(halves))


def test_to_u8_clamps_out_of_range_alpha(cpu, gpu, rng):
    alpha = (rng.random((H, W), np.float32) * 3.0 - 1.0).astype(np.float32)
    np.testing.assert_array_equal(gpu.download(gpu.to_u8(gpu.upload(alpha))), cpu.to_u8(alpha))


# --------------------------------------------------------------------------- elementwise

def test_over_127_matches(cpu, gpu, rng):
    frame = u8(rng)
    np.testing.assert_array_equal(gpu.download(gpu.over_127(gpu.upload(frame))),
                                  cpu.over_127(frame))


def test_or_into_accumulates_the_same(cpu, gpu, rng):
    frames = [u8(rng) for _ in range(5)]
    want = None
    got = None
    for frame in frames:
        want = cpu.or_into(want, frame)
        got = gpu.or_into(got, gpu.upload(frame))
    np.testing.assert_array_equal(gpu.download(got), want)


def test_or_into_does_not_write_through_to_its_first_argument(cpu, gpu, rng):
    """The accumulator is the backend's own buffer, not the caller's frame.

    `or_into(None, x)` must copy: `x` is a frame still sitting in the prior's window, and
    or-ing into it would corrupt every later use of it.
    """
    first, second = u8(rng), u8(rng)
    source = gpu.upload(first)
    acc = gpu.or_into(None, source)
    gpu.or_into(acc, gpu.upload(second))
    np.testing.assert_array_equal(gpu.download(source), first)


def test_select_matches(cpu, gpu, rng):
    frame = u8(rng)
    keep = np.where(rng.random((H, W)) > 0.5, np.uint8(255), np.uint8(0))
    np.testing.assert_array_equal(
        gpu.download(gpu.select(gpu.upload(frame), gpu.upload(keep))),
        cpu.select(frame, keep))


@pytest.mark.parametrize("size", [3, 7, 15])
def test_dilate_matches_including_at_the_border(cpu, gpu, rng, size):
    frame = u8(rng)
    frame[:2, :] = 255  # morphology border handling is where the two could disagree
    frame[-2:, :] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    np.testing.assert_array_equal(gpu.download(gpu.dilate(gpu.upload(frame), kernel)),
                                  cpu.dilate(frame, kernel))


def test_scale_by_matches(cpu, gpu, rng):
    shape, level = u8(rng), u8(rng)
    np.testing.assert_array_equal(
        gpu.download(gpu.scale_by(gpu.upload(shape), gpu.upload(level))),
        cpu.scale_by(shape, level))


def test_peak_and_any_agree(cpu, gpu, rng):
    frame = u8(rng)
    assert gpu.peak(gpu.upload(frame)) == cpu.peak(frame)
    assert gpu.any(gpu.upload(frame)) == cpu.any(frame)
    empty = np.zeros((H, W), np.uint8)
    assert gpu.any(gpu.upload(empty)) == cpu.any(empty) is False


# --------------------------------------------------------------------------- temporal

@pytest.mark.parametrize("count", [2, 3, 4, 5, 7])
def test_median_matches_at_every_window_length(cpu, gpu, rng, count):
    """Even lengths included: those are the truncated windows at a clip's first and last
    frames, and they take a different route through the backend."""
    window = [u8(rng) for _ in range(count)]
    np.testing.assert_array_equal(
        gpu.download(gpu.median([gpu.upload(f) for f in window])),
        cpu.median(window))


def test_max_reduce_matches(cpu, gpu, rng):
    window = [u8(rng) for _ in range(4)]
    np.testing.assert_array_equal(
        gpu.download(gpu.max_reduce([gpu.upload(f) for f in window])),
        cpu.max_reduce(window))


def test_median_does_not_disturb_its_inputs(gpu, rng):
    """The window holds frames the prior still needs; sorting must happen in scratch."""
    window = [u8(rng) for _ in range(5)]
    uploaded = [gpu.upload(f) for f in window]
    gpu.median(uploaded)
    for original, buf in zip(window, uploaded):
        np.testing.assert_array_equal(gpu.download(buf), original)


# --------------------------------------------------------------------------- pooling

def test_a_retained_buffer_is_not_recycled_while_it_is_still_held(cpu, gpu, rng):
    """The failure this guards against is silent and frame-dependent.

    Buffers are pooled because allocating one costs fifty times what the kernel does. That is
    only sound while the ring is longer than the window holding its results - get it wrong and
    a mask still sitting in the prior's window is quietly overwritten by a later frame's, with
    no error anywhere.
    """
    depth = 26
    sources = [(np.full((H, W), i % 256, np.float32) / 255.0) for i in range(depth + 10)]
    held = [gpu.to_u8(gpu.upload(src), "level") for src in sources]
    for src, buf in list(zip(sources, held))[-depth:]:
        np.testing.assert_array_equal(gpu.download(buf), cpu.to_u8(src))


def test_reserve_grows_a_ring_and_never_shrinks_it(gpu):
    gpu.reserve({"level": 40})
    gpu.reserve({"level": 4})
    frame = np.zeros((H, W), np.float32)
    held = [gpu.to_u8(gpu.upload(frame), "level") for _ in range(40)]
    assert len({id(buf) for buf in held}) == 40


def test_a_clip_of_a_different_size_does_not_leave_its_buffers_behind(gpu, cpu):
    """A folder of mixed resolutions must not accumulate a buffer set per resolution."""
    first = np.zeros((H, W), np.float32)
    gpu.to_u8(gpu.upload(first), "level")
    assert gpu._rings

    second = np.zeros((H // 2, W // 2), np.float32)
    got = gpu.to_u8(gpu.upload(second), "level")

    assert all(key[1:3] == (H // 2, W // 2) for key in gpu._rings), \
        "rings for the previous clip size were kept"
    np.testing.assert_array_equal(gpu.download(got), cpu.to_u8(second))


def test_the_backend_still_works_after_the_size_changes_back(gpu, cpu, rng):
    """Clearing the rings must not leave the pool in a state that hands out stale buffers."""
    for shape in ((H, W), (H // 2, W // 2), (H, W)):
        alpha = rng.random(shape, np.float32)
        np.testing.assert_array_equal(gpu.download(gpu.to_u8(gpu.upload(alpha), "level")),
                                      cpu.to_u8(alpha))


def test_a_cpu_only_opencv_is_reported_rather_than_silently_absorbed(monkeypatch):
    """The failure mode this guards is a scan that is quietly half as fast.

    A `pip install` that pulls a stock OpenCV over the CUDA one leaves no other trace, so the
    fallback has to say something.
    """
    import dsf.accel as accel_mod

    monkeypatch.delenv("DSF_ACCEL", raising=False)
    monkeypatch.setattr(accel_mod, "cuda_available", lambda: False)
    monkeypatch.setattr("dsf.detect.base.resolve_device", lambda _: "cuda")
    accel_mod._warn_no_cuda_opencv.cache_clear()

    with pytest.warns(RuntimeWarning, match="install_opencv_cuda"):
        assert accel_mod.resolve("auto") == "cpu"
    accel_mod._warn_no_cuda_opencv.cache_clear()


def test_no_gpu_at_all_is_not_worth_warning_about(monkeypatch):
    """A machine without an NVIDIA card is not a broken install."""
    import dsf.accel as accel_mod

    monkeypatch.delenv("DSF_ACCEL", raising=False)
    monkeypatch.setattr(accel_mod, "cuda_available", lambda: False)
    monkeypatch.setattr("dsf.detect.base.resolve_device", lambda _: "cpu")
    accel_mod._warn_no_cuda_opencv.cache_clear()

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert accel_mod.resolve("auto") == "cpu"
