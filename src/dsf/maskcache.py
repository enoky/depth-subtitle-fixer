"""On-disk cache of the computed alpha masks.

Detection is the expensive half of the pipeline and brightness is the knob most likely to be
re-tuned, so the two are decoupled: ``dsf detect`` writes the mask once, and ``dsf render``
(and the Gradio brightness slider) replays it. Masks are stored as a lossless 8-bit grayscale
video, which is compact, scrubbable, and viewable in any player.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np

from . import __version__
from .videoio import GrayWriter, read_gray


def sidecar_path(mask_path: str | Path) -> Path:
    return Path(str(mask_path) + ".json")


def source_fingerprint(path: str | Path) -> str:
    """Cheap identity for a media file: size + mtime + hash of the first megabyte."""
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(str(st.st_size).encode())
    h.update(str(int(st.st_mtime)).encode())
    with p.open("rb") as fh:
        h.update(fh.read(1 << 20))
    return h.hexdigest()[:16]


@dataclass
class MaskMeta:
    width: int
    height: int
    fps: str
    frames: int
    rgb_path: str
    rgb_fingerprint: str
    config: dict
    version: str = __version__

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MaskMeta":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data.pop("version", None)
        return cls(**data, version=__version__)


class MaskCacheWriter:
    """Write alpha masks plus a sidecar describing how they were produced."""

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction,
                 rgb_path: str, config: dict):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = GrayWriter(self.path, width, height, fps)
        self.width, self.height, self.fps = width, height, fps
        self.rgb_path = str(rgb_path)
        self.config = config
        self.frames = 0

    def write(self, mask_u8: np.ndarray) -> None:
        self.writer.write(mask_u8)
        self.frames += 1

    def close(self) -> None:
        self.writer.close()
        meta = MaskMeta(
            width=self.width, height=self.height,
            fps=f"{self.fps.numerator}/{self.fps.denominator}",
            frames=self.frames, rgb_path=self.rgb_path,
            rgb_fingerprint=source_fingerprint(self.rgb_path),
            config=self.config,
        )
        sidecar_path(self.path).write_text(meta.to_json(), encoding="utf-8")

    def __enter__(self) -> "MaskCacheWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_masks(path: str | Path) -> Iterator[np.ndarray]:
    """Yield uint8 HxW alpha masks from a cache file."""
    yield from read_gray(path)


def load_meta(path: str | Path) -> MaskMeta | None:
    sidecar = sidecar_path(path)
    if not sidecar.exists():
        return None
    try:
        return MaskMeta.load(sidecar)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def cache_matches(mask_path: str | Path, rgb_path: str | Path) -> bool:
    """True when the cache was built from this exact RGB file."""
    meta = load_meta(mask_path)
    if meta is None:
        return False
    try:
        return meta.rgb_fingerprint == source_fingerprint(rgb_path)
    except OSError:
        return False
