"""PNG/TIFF image-sequence input and output.

DepthCrafter is often driven and reviewed as frames rather than as a movie, so a folder of
stills is a first-class input here, not a conversion step.

Read directly with OpenCV rather than through ffmpeg. A sequence needs no demuxing, and
going via a video pipeline would force an 8-bit or YUV round trip that a 16-bit depth PNG
would not survive. Reading the files means the stored values arrive exactly as written, and
anything the mask does not touch can be handed back byte for byte.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np

from .videoio import VideoInfo

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".ppm", ".pgm")
_DIGITS = re.compile(r"(\d+)")


def is_sequence(path: str | Path) -> bool:
    """True when *path* is a directory holding at least one readable still."""
    p = Path(path)
    return p.is_dir() and bool(frame_paths(p))


def frame_paths(path: str | Path) -> list[Path]:
    """Every image in the folder, ordered by the number in its name."""
    p = Path(path)
    if not p.is_dir():
        return []
    files = [f for f in p.iterdir()
             if f.is_file() and f.suffix.lower() in IMAGE_SUFFIXES]

    def key(f: Path):
        digits = _DIGITS.findall(f.stem)
        # Sort on the trailing number so frame9 precedes frame10, with the name as a
        # tie-break for anything unnumbered.
        return (int(digits[-1]) if digits else -1, f.name.lower())

    return sorted(files, key=key)


def _read(path: Path) -> np.ndarray:
    """Load a still exactly as stored - bit depth and channel count preserved."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise OSError(f"could not read image: {path}")
    return image


def sequence_info(path: str | Path, fps: float = 24.0) -> VideoInfo:
    """Describe a folder of stills the same way probe() describes a video."""
    files = frame_paths(path)
    if not files:
        raise FileNotFoundError(f"no images found in {path}")
    first = _read(files[0])
    height, width = first.shape[:2]
    bit_depth = 16 if first.dtype == np.uint16 else 8
    channels = 1 if first.ndim == 2 else first.shape[2]
    return VideoInfo(
        path=str(path),
        width=int(width),
        height=int(height),
        pix_fmt=f"{'gray' if channels == 1 else 'rgb'}{bit_depth}",
        codec_name="imageseq",
        fps=Fraction(fps).limit_denominator(1000),
        nb_frames=len(files),
        frames_exact=True,
        # Stills are always full range; there is no limited-range convention for a PNG.
        color_range="pc",
        color_primaries=None,
        color_transfer=None,
        color_space=None,
        bit_depth=bit_depth,
        chroma="400" if channels == 1 else "444",
    )


def read_rgb(path: str | Path, start: int = 0,
             max_frames: int | None = None) -> Iterator[np.ndarray]:
    """Yield uint8 HxWx3 RGB frames, matching the video reader's contract."""
    files = frame_paths(path)[start:]
    if max_frames is not None:
        files = files[:max_frames]
    for f in files:
        image = _read(f)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = image[..., :3]
        if image.dtype == np.uint16:
            image = (image >> 8).astype(np.uint8)
        yield cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@dataclass
class DepthStill:
    """One depth frame plus the filename it came from, so output can mirror input."""

    name: str
    image: np.ndarray  # exactly as stored: 2D or HxWxC, uint8 or uint16

    @property
    def plane(self) -> np.ndarray:
        """The depth values as a single uint16 plane.

        Depth stills are written as grey but often land as RGB with a code or two of drift
        between channels from whatever encoded them. Averaging is steadier than picking one
        channel, and the drift is put back on write.
        """
        if self.image.ndim == 2:
            return self.image.astype(np.uint16)
        return np.rint(self.image.mean(axis=2)).astype(np.uint16)

    def with_plane(self, new_plane: np.ndarray) -> np.ndarray:
        """Reapply an edited depth plane, keeping untouched pixels bit-exact.

        Written as a *difference* rather than a replacement: where the mask did nothing the
        delta is zero and the original bytes survive, including any per-channel drift, and
        where it did the shift lands on every channel equally.
        """
        maximum = 65535 if self.image.dtype == np.uint16 else 255
        delta = new_plane.astype(np.int32) - self.plane.astype(np.int32)
        if self.image.ndim == 3:
            delta = delta[..., None]
        out = self.image.astype(np.int32) + delta
        return np.clip(out, 0, maximum).astype(self.image.dtype)


def read_depth(path: str | Path, start: int = 0,
               max_frames: int | None = None) -> Iterator[DepthStill]:
    files = frame_paths(path)[start:]
    if max_frames is not None:
        files = files[:max_frames]
    for f in files:
        yield DepthStill(name=f.name, image=_read(f))


class SequenceWriter:
    """Write stills into a folder, reusing each source frame's filename."""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames = 0

    def write(self, name: str, image: np.ndarray) -> None:
        target = self.out_dir / name
        if target.suffix.lower() not in IMAGE_SUFFIXES:
            target = target.with_suffix(".png")
        if not cv2.imwrite(str(target), image):
            raise OSError(f"could not write image: {target}")
        self.frames += 1

    def close(self) -> None:  # symmetry with DepthWriter
        pass

    def __enter__(self) -> "SequenceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def looks_like_output_sequence(path: str | Path) -> bool:
    """An output path with no file extension is a folder of frames."""
    return Path(path).suffix == ""


def sample(path: str | Path, indices: Sequence[int]) -> dict[int, Path]:
    files = frame_paths(path)
    return {i: files[i] for i in indices if 0 <= i < len(files)}
