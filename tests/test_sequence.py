"""PNG/TIFF sequence input and output."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from dsf import media, sequence


def write_stills(folder, frames, name="frame_{:04d}.png"):
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        assert cv2.imwrite(str(folder / name.format(i)), f)
    return folder


def test_a_folder_of_stills_is_recognised(tmp_path):
    folder = write_stills(tmp_path / "seq", [np.zeros((8, 12, 3), np.uint8)] * 3)
    assert sequence.is_sequence(folder)
    assert media.is_sequence(folder)


def test_an_empty_folder_is_not_a_sequence(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert not sequence.is_sequence(empty)


def test_frames_are_ordered_by_number_not_by_string(tmp_path):
    folder = tmp_path / "seq"
    folder.mkdir()
    for n in (1, 2, 10, 20, 100):
        cv2.imwrite(str(folder / f"f{n}.png"), np.zeros((4, 4), np.uint8))
    names = [p.stem for p in sequence.frame_paths(folder)]
    assert names == ["f1", "f2", "f10", "f20", "f100"], "f10 must not sort before f2"


def test_sequence_info_matches_the_stills(tmp_path):
    folder = write_stills(tmp_path / "seq", [np.zeros((40, 60, 3), np.uint8)] * 5)
    info = sequence.sequence_info(folder, fps=25)
    assert (info.width, info.height) == (60, 40)
    assert info.nb_frames == 5 and info.frames_exact
    assert info.bit_depth == 8
    assert info.color_range == "pc", "stills are always full range"
    assert float(info.fps) == pytest.approx(25.0)


def test_sixteen_bit_stills_are_reported_as_such(tmp_path):
    folder = write_stills(tmp_path / "seq", [np.zeros((8, 8), np.uint16)] * 2)
    info = sequence.sequence_info(folder)
    assert info.bit_depth == 16
    assert info.chroma == "400"


def test_rgb_reader_yields_uint8_rgb(tmp_path):
    # OpenCV writes BGR, so a pure-red frame proves the channel order comes back right.
    bgr = np.zeros((6, 6, 3), np.uint8)
    bgr[..., 2] = 255
    folder = write_stills(tmp_path / "seq", [bgr] * 2)
    frames = list(sequence.read_rgb(folder))
    assert len(frames) == 2
    assert frames[0].dtype == np.uint8 and frames[0].shape == (6, 6, 3)
    assert tuple(frames[0][0, 0]) == (255, 0, 0), "expected RGB order"


def test_reader_honours_start_and_count(tmp_path):
    frames = [np.full((4, 4), i * 10, np.uint8) for i in range(6)]
    folder = write_stills(tmp_path / "seq", frames)
    got = list(sequence.read_depth(folder, start=2, max_frames=3))
    assert [int(g.image[0, 0]) for g in got] == [20, 30, 40]


def test_depth_plane_averages_a_near_grey_still():
    image = np.dstack([np.full((4, 4), 100, np.uint8),
                       np.full((4, 4), 102, np.uint8),
                       np.full((4, 4), 101, np.uint8)])
    still = sequence.DepthStill("f.png", image)
    assert int(still.plane[0, 0]) == 101


def test_untouched_pixels_survive_byte_for_byte():
    """The whole reason edits are applied as a difference rather than a replacement.

    Built like a real depth still: grey, with a code or two of drift between channels from
    whatever encoded it. That drift is exactly what a replacement would throw away.
    """
    rng = np.random.default_rng(0)
    grey = rng.integers(40, 160, (16, 16), dtype=np.uint8).astype(np.int16)
    tint = rng.integers(-2, 3, (16, 16, 3))
    image = np.clip(grey[..., None] + tint, 0, 255).astype(np.uint8)

    still = sequence.DepthStill("f.png", image)
    plane = still.plane.copy()
    plane[4:8, 4:8] = 250  # edit one patch
    out = still.with_plane(plane)

    untouched = np.ones((16, 16), bool)
    untouched[4:8, 4:8] = False
    np.testing.assert_array_equal(out[untouched], image[untouched])
    assert out.dtype == image.dtype and out.shape == image.shape
    assert int(out[5, 5].mean()) == pytest.approx(250, abs=1)
    # The per-channel drift rides along with the edit rather than being flattened out.
    np.testing.assert_array_equal(out[5, 5].astype(np.int16) - int(out[5, 5].mean().round()),
                                  image[5, 5].astype(np.int16) - int(still.plane[5, 5]))


def test_sixteen_bit_depth_survives_a_round_trip(tmp_path):
    """A 16-bit depth still must not be quantised on the way through."""
    image = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    folder = write_stills(tmp_path / "seq", [image])
    still = next(iter(sequence.read_depth(folder)))
    assert still.image.dtype == np.uint16
    np.testing.assert_array_equal(still.image, image)
    np.testing.assert_array_equal(still.with_plane(still.plane), image)


def test_writer_reuses_the_source_filenames(tmp_path):
    frames = [np.full((4, 4), i, np.uint8) for i in range(3)]
    src = write_stills(tmp_path / "in", frames, name="{:08d}.png")
    out = tmp_path / "out"
    with sequence.SequenceWriter(out) as writer:
        for still in sequence.read_depth(src):
            writer.write(still.name, still.image)
    assert [p.name for p in sequence.frame_paths(out)] == \
           [p.name for p in sequence.frame_paths(src)]


def test_media_probe_dispatches_to_the_sequence(tmp_path):
    folder = write_stills(tmp_path / "seq", [np.zeros((20, 30, 3), np.uint8)] * 4)
    info = media.probe(folder)
    assert info.codec_name == "imageseq"
    assert info.nb_frames == 4


def test_a_sequence_cannot_be_written_into_a_video_file(tmp_path):
    """Better to say so than to silently transcode a frame workflow into an mp4."""
    from dsf.config import PipelineConfig

    folder = write_stills(tmp_path / "seq", [np.zeros((8, 8, 3), np.uint8)])
    info = media.probe(folder)
    with pytest.raises(ValueError, match="output folder"):
        media.open_depth_sink(str(tmp_path / "out.mp4"), info, PipelineConfig(), True)


def test_extensionless_output_means_a_folder(tmp_path):
    from dsf.config import PipelineConfig

    folder = write_stills(tmp_path / "seq", [np.zeros((8, 8, 3), np.uint8)])
    info = media.probe(folder)
    sink = media.open_depth_sink(str(tmp_path / "out"), info, PipelineConfig(), True)
    assert isinstance(sink, media._SequenceSink)
    sink.close()
