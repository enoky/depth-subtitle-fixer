"""Video I/O is the part that must not lose a single code value."""

from __future__ import annotations

import numpy as np
import pytest

from dsf.videoio import (
    DepthFrame, DepthWriter, GrayWriter, probe, read_depth, read_gray, synth_test_video,
)


def ramp_frames(width=160, height=96, count=5, bit_depth=10):
    """Frames spanning the legal luma range, plus a per-frame offset."""
    lo, hi = 16 << (bit_depth - 8), 235 << (bit_depth - 8)
    base = np.linspace(lo, hi, width, dtype=np.float32)[None, :].repeat(height, axis=0)
    return [np.clip(base + i * 3, lo, hi).astype(np.uint16) for i in range(count)]


def test_probe_reads_10bit_metadata(tmp_path):
    path = tmp_path / "ramp.mp4"
    synth_test_video(path, ramp_frames(), fps=24, color_range="tv")
    info = probe(path)

    assert (info.width, info.height) == (160, 96)
    assert info.bit_depth == 10
    assert info.chroma == "420"
    assert info.decode_pix_fmt == "yuv420p10le"
    assert info.color_range == "tv"
    assert float(info.fps) == pytest.approx(24.0)
    assert info.frame_nbytes == (160 * 96 + 2 * 80 * 48) * 2


def test_decode_is_bit_exact(tmp_path):
    """A lossless encode must survive the round trip without a single altered code."""
    frames = ramp_frames(count=4)
    path = tmp_path / "ramp.mp4"
    synth_test_video(path, frames, lossless=True)

    decoded = [f.y for f in read_depth(path)]
    assert len(decoded) == len(frames)
    for original, got in zip(frames, decoded):
        assert got.dtype == np.uint16
        np.testing.assert_array_equal(got, original)


def test_writer_roundtrip_preserves_untouched_pixels(tmp_path):
    """Only the pixels we deliberately change may differ after a lossless re-encode."""
    frames = ramp_frames(count=3)
    src = tmp_path / "src.mp4"
    synth_test_video(src, frames, lossless=True)
    info = probe(src)

    out = tmp_path / "out.mp4"
    edited = []
    with DepthWriter(out, info, encoder="libx265", lossless=True) as writer:
        for frame in read_depth(src, info):
            y = frame.y.copy()
            y[10:20, 10:20] = 900  # a deliberate edit
            edited.append(y)
            writer.write(DepthFrame(y=y, u=frame.u, v=frame.v))

    decoded = [f.y for f in read_depth(out)]
    assert len(decoded) == len(edited)
    for want, got in zip(edited, decoded):
        np.testing.assert_array_equal(got, want)


def test_writer_preserves_stream_properties(tmp_path):
    frames = ramp_frames(count=3)
    src = tmp_path / "src.mp4"
    synth_test_video(src, frames, fps=30, color_range="tv")
    info = probe(src)

    out = tmp_path / "out.mp4"
    with DepthWriter(out, info, crf=12) as writer:
        for frame in read_depth(src, info):
            writer.write(frame)

    got = probe(out)
    assert (got.width, got.height) == (info.width, info.height)
    assert got.bit_depth == info.bit_depth
    assert got.chroma == info.chroma
    assert got.color_range == info.color_range
    assert float(got.fps) == pytest.approx(float(info.fps))


def test_full_range_source_is_not_rescaled(tmp_path):
    """A pc-range depth map must come back with its original codes, not squeezed to 64-940."""
    frames = [np.full((64, 64), 1000, dtype=np.uint16),
              np.full((64, 64), 5, dtype=np.uint16)]
    path = tmp_path / "full.mp4"
    synth_test_video(path, frames, color_range="pc", lossless=True)

    info = probe(path)
    assert info.color_range == "pc"
    decoded = [f.y for f in read_depth(path, info)]
    assert int(decoded[0].max()) == 1000
    assert int(decoded[1].min()) == 5


def test_gray_writer_is_lossless(tmp_path):
    rng = np.random.default_rng(0)
    masks = [rng.integers(0, 256, (48, 64), dtype=np.uint8) for _ in range(4)]
    path = tmp_path / "mask.mkv"
    with GrayWriter(path, 64, 48, __import__("fractions").Fraction(24)) as writer:
        for mask in masks:
            writer.write(mask)

    decoded = list(read_gray(path))
    assert len(decoded) == len(masks)
    for want, got in zip(masks, decoded):
        np.testing.assert_array_equal(got, want)


def test_seek_lands_on_the_requested_frame(tmp_path):
    """Each frame gets a unique flat value so we can identify it after a seek."""
    frames = [np.full((64, 64), 100 + i * 20, dtype=np.uint16) for i in range(12)]
    path = tmp_path / "counted.mp4"
    synth_test_video(path, frames, lossless=True)
    info = probe(path)

    for target in (0, 3, 7, 11):
        first = next(iter(read_depth(path, info, seek_frame=target)))
        assert int(first.y[0, 0]) == 100 + target * 20
