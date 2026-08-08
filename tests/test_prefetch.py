"""Reading ahead of the consumer, so decode and inference stop taking turns."""

from __future__ import annotations

import threading
import time

import pytest

from dsf.prefetch import DEFAULT_BUDGET, depth_for, prefetch


def test_everything_arrives_in_order():
    assert list(prefetch(iter(range(200)))) == list(range(200))


def test_an_empty_stream_is_fine():
    assert list(prefetch(iter([]))) == []


def test_exiting_is_prompt_whether_or_not_the_stream_ran_out():
    """Regression: waiting out a drain timeout per cluster turned a 25s scan into 265s.

    A stream that ends by itself has nothing left to drain, and blocking on the empty queue
    to discover that costs the whole timeout every time.
    """
    for consume_all in (True, False):
        started = time.monotonic()
        stream = prefetch(iter(range(20)))
        if consume_all:
            list(stream)
        else:
            next(stream)
        stream.close()
        assert time.monotonic() - started < 1.0, f"slow exit (consume_all={consume_all})"


def test_the_reader_is_shut_down_when_the_consumer_stops_early():
    """Every clip that has seen enough stops early, leaving ffmpeg running if this leaks."""
    closed = threading.Event()

    def source():
        try:
            for i in range(10_000):
                yield i
        finally:
            closed.set()

    stream = prefetch(source())
    assert [next(stream) for _ in range(3)] == [0, 1, 2]
    stream.close()
    assert closed.wait(timeout=5), "the reader was left running"


def test_a_failure_surfaces_on_the_consumers_thread():
    """A decode failure must reach the caller, not vanish into a worker."""
    def source():
        yield 1
        raise OSError("ffmpeg died mid-clip")

    with pytest.raises(OSError, match="ffmpeg died"):
        list(prefetch(source()))


def test_no_more_than_the_depth_is_held():
    """Frames are uncompressed; an unbounded queue would be a memory bomb at 4K."""
    handed_out: list[int] = []

    def source():
        for i in range(500):
            handed_out.append(i)
            yield i

    stream = prefetch(source(), depth=2)
    next(stream)
    time.sleep(0.2)  # let the reader run as far ahead as it is allowed to
    assert len(handed_out) <= 6, handed_out
    stream.close()


def test_depth_is_sized_against_the_frame_not_the_frame_count():
    """A depth in frames means something very different at 720p and at 8K."""
    assert depth_for(1920, 1080, wanted=8) == 8
    assert depth_for(7680, 4320, wanted=8) < 8
    assert depth_for(7680, 4320, wanted=8) >= 1
    # Never more than asked for, however small the frames are.
    assert depth_for(64, 64, wanted=3) == 3
    assert depth_for(1920, 1080, wanted=8, budget=DEFAULT_BUDGET) == 8
