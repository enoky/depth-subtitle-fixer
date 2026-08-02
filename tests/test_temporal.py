"""Temporal mask filtering."""

from __future__ import annotations

import numpy as np
import pytest

from dsf.config import TemporalConfig
from dsf.temporal import from_u8, smooth, to_u8


def mask(value: int, shape=(8, 8)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


def test_u8_roundtrip():
    alpha = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(from_u8(to_u8(alpha)), alpha, atol=1 / 255)


def test_median_removes_a_single_frame_false_positive():
    window = [mask(0), mask(0), mask(255), mask(0), mask(0)]
    out = smooth(window[2], window, TemporalConfig(mode="median"))
    assert int(out.max()) == 0


def test_median_fills_a_single_frame_miss():
    window = [mask(255), mask(255), mask(0), mask(255), mask(255)]
    out = smooth(window[2], window, TemporalConfig(mode="median"))
    assert int(out.min()) == 255


def test_max_keeps_anything_seen_in_the_window():
    window = [mask(0), mask(200), mask(0)]
    out = smooth(window[1], window, TemporalConfig(mode="max"))
    assert int(out.max()) == 200


def test_none_returns_the_centre_untouched():
    window = [mask(0), mask(123), mask(255)]
    out = smooth(window[1], window, TemporalConfig(mode="none"))
    assert int(out[0, 0]) == 123


def test_centre_is_used_not_the_middle_of_a_truncated_window():
    """Windows are clipped at the clip boundaries, so position != centre."""
    window = [mask(10), mask(20), mask(30)]
    out = smooth(window[0], window, TemporalConfig(mode="none"))
    assert int(out[0, 0]) == 10


def test_single_frame_window_is_a_passthrough():
    out = smooth(mask(77), [mask(77)], TemporalConfig(mode="median"))
    assert int(out[0, 0]) == 77


@pytest.mark.parametrize("count", range(2, 13))
def test_the_sorting_network_really_sorts(count):
    """By the 0-1 principle: a comparator network that sorts every binary input sorts any.

    The network is generated for the next power of two with the comparators reaching past
    the end dropped, and that it still sorts is the whole basis for not padding the window
    out - padding would put invented values in the middle, where the median is read from.
    """
    import itertools

    from dsf.temporal import _sorting_network

    network = _sorting_network(count)
    for bits in itertools.product((0, 1), repeat=count):
        values = list(bits)
        for lo, hi in network:
            if values[lo] > values[hi]:
                values[lo], values[hi] = values[hi], values[lo]
        assert values == sorted(values), f"{count} items, {bits} came out {values}"


@pytest.mark.parametrize("count", range(2, 10))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_median_matches_numpy_including_the_even_case(count, seed):
    """The masks this produces go into a depth map, so "faster" has to mean "identical".

    Even windows are the interesting ones: they happen when a window is truncated at a clip
    boundary, and they are the only case where the middle pair is averaged.
    """
    rng = np.random.default_rng(seed)
    window = [rng.integers(0, 256, (23, 31), dtype=np.uint8) for _ in range(count)]
    if seed == 1:  # lots of ties, where an off-by-one in the middle pair would hide
        window = [rng.integers(0, 3, (23, 31), dtype=np.uint8) for _ in range(count)]

    want = np.median(np.stack(window, axis=0), axis=0).astype(np.uint8)
    got = smooth(window[count // 2], window, TemporalConfig(mode="median"))
    assert got.dtype == np.uint8
    assert np.array_equal(got, want)


def test_smoothing_does_not_write_to_the_frames_it_was_given():
    """The window is a shared buffer - the same arrays are the next frame's window too."""
    rng = np.random.default_rng(3)
    window = [rng.integers(0, 256, (16, 16), dtype=np.uint8) for _ in range(5)]
    before = [frame.copy() for frame in window]

    for mode in ("median", "max"):
        smooth(window[2], window, TemporalConfig(mode=mode))
        for original, now in zip(before, window):
            assert np.array_equal(original, now), f"{mode} modified its input"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        smooth(mask(0), [mask(0), mask(1)], TemporalConfig(mode="bogus"))
