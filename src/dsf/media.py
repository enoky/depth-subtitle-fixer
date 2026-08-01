"""One way in and out, whether the media is a movie or a folder of frames.

Everything above this module works in terms of "a depth plane to edit"; only this layer
knows whether that plane came out of a 10-bit video stream or an 8-bit PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from . import sequence, videoio
from .videoio import DepthFrame, VideoInfo


def is_sequence(path: str | Path) -> bool:
    return sequence.is_sequence(path)


def probe(path: str | Path, fps: float = 24.0) -> VideoInfo:
    if sequence.is_sequence(path):
        return sequence.sequence_info(path, fps=fps)
    return videoio.probe(path)


def read_rgb(path: str | Path, start: int = 0, seek_frame: int = 0,
             max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield uint8 HxWx3 RGB frames.

    A sequence has no notion of seeking - skipping files *is* the seek - so the two
    positioning arguments collapse into one offset.
    """
    if sequence.is_sequence(path):
        yield from sequence.read_rgb(path, start=start + seek_frame, max_frames=max_frames)
        return
    count = 0
    for frame in videoio.read_rgb(path, start=start, seek_frame=seek_frame):
        yield frame
        count += 1
        if max_frames is not None and count >= max_frames:
            return


def read_depth(path: str | Path, info: VideoInfo | None = None, start: int = 0,
               seek_frame: int = 0) -> Iterator:
    """Yield depth units - DepthFrame for video, DepthStill for a sequence.

    Both expose ``.plane`` as a uint16 2D array, which is all the compositor needs.
    """
    if sequence.is_sequence(path):
        yield from sequence.read_depth(path, start=start + seek_frame)
        return
    yield from videoio.read_depth(path, info, start=start, seek_frame=seek_frame)


class _VideoSink:
    def __init__(self, out_path: str, info: VideoInfo, cfg):
        self.writer = videoio.DepthWriter(
            out_path, info, encoder=cfg.encode.encoder, crf=cfg.encode.crf,
            preset=cfg.encode.preset, lossless=cfg.encode.lossless,
        )
        self.frames = 0

    def write(self, unit: DepthFrame, plane: np.ndarray) -> None:
        self.writer.write(DepthFrame(y=plane, u=unit.u, v=unit.v))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()


class _SequenceSink:
    def __init__(self, out_dir: str):
        self.writer = sequence.SequenceWriter(out_dir)
        self.frames = 0

    def write(self, unit: "sequence.DepthStill", plane: np.ndarray) -> None:
        self.writer.write(unit.name, unit.with_plane(plane))
        self.frames += 1

    def close(self) -> None:
        self.writer.close()


def open_depth_sink(out_path: str | Path, info: VideoInfo, cfg, source_is_sequence: bool):
    """Pick an output form: a folder of frames in, a folder of frames out.

    An output path with no extension also means a folder, so a sequence can be rendered to
    a video and vice versa without a separate flag.
    """
    out_path = str(out_path)
    wants_sequence = sequence.looks_like_output_sequence(out_path) or \
        (source_is_sequence and Path(out_path).suffix == "")
    if wants_sequence:
        return _SequenceSink(out_path)
    if source_is_sequence:
        raise ValueError(
            f"cannot encode a PNG sequence into {out_path!r}: give an output folder "
            f"(a path with no extension) so the frames are written back as stills"
        )
    return _VideoSink(out_path, info, cfg)
