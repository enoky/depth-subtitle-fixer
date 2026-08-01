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


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        smooth(mask(0), [mask(0), mask(1)], TemporalConfig(mode="bogus"))
