"""ffmpeg/ffprobe based video I/O.

The depth path is precision-critical: DepthCrafter output is a 10-bit luma plane and we must
not let swscale touch it. So the depth reader decodes to the *source* pixel format and slices
the planes ourselves, and the writer feeds raw planes straight back to the encoder. Only the
pixels under the text mask are ever modified; chroma planes are passed through untouched.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

_PIXFMT_RE = re.compile(
    r"^(?:yuvj?(?P<chroma>4[0-4][0-4])p(?P<depth>\d+)?(?:le|be)?"
    r"|(?P<gray>gray)(?P<gdepth>\d+)?(?:le|be)?)$"
)


class FFmpegError(RuntimeError):
    pass


def _resolve(name: str) -> str:
    """Find ffmpeg/ffprobe, falling back to the imageio-ffmpeg bundled binary."""
    found = shutil.which(name)
    if found:
        return found
    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # pragma: no cover - only hit without any ffmpeg at all
            pass
    raise FFmpegError(
        f"could not find {name} on PATH. Install ffmpeg (https://ffmpeg.org) or "
        f"`pip install imageio-ffmpeg`."
    )


def ffmpeg_exe() -> str:
    return _resolve("ffmpeg")


def ffprobe_exe() -> str:
    return _resolve("ffprobe")


@dataclass(frozen=True)
class VideoInfo:
    path: str
    width: int
    height: int
    pix_fmt: str
    codec_name: str
    fps: Fraction
    nb_frames: int
    frames_exact: bool
    color_range: str  # "tv" | "pc"
    color_primaries: str | None
    color_transfer: str | None
    color_space: str | None
    bit_depth: int
    chroma: str  # "400" | "420" | "422" | "444"

    @property
    def decode_pix_fmt(self) -> str:
        """Canonical little-endian pixel format to request from ffmpeg for raw decode."""
        if self.chroma == "400":
            return "gray" if self.bit_depth == 8 else f"gray{self.bit_depth}le"
        if self.bit_depth == 8:
            return f"yuv{self.chroma}p"
        return f"yuv{self.chroma}p{self.bit_depth}le"

    @property
    def max_code(self) -> int:
        return (1 << self.bit_depth) - 1

    def plane_shapes(self) -> list[tuple[int, int]]:
        shapes = [(self.height, self.width)]
        if self.chroma == "400":
            return shapes
        if self.chroma == "420":
            cw, ch = (self.width + 1) // 2, (self.height + 1) // 2
        elif self.chroma == "422":
            cw, ch = (self.width + 1) // 2, self.height
        else:
            cw, ch = self.width, self.height
        shapes += [(ch, cw), (ch, cw)]
        return shapes

    @property
    def frame_nbytes(self) -> int:
        bps = 1 if self.bit_depth == 8 else 2
        return sum(h * w for h, w in self.plane_shapes()) * bps


def _parse_pix_fmt(pix_fmt: str, bits_per_raw_sample: int | None) -> tuple[str, int]:
    m = _PIXFMT_RE.match(pix_fmt or "")
    if not m:
        # Unknown/exotic format: decode through a well-defined 10-bit 4:2:0 intermediate.
        return "420", 10
    if m.group("gray"):
        depth = int(m.group("gdepth") or 8)
        return "400", depth
    chroma = m.group("chroma")
    depth = int(m.group("depth") or bits_per_raw_sample or 8)
    return chroma, depth


def probe(path: str | Path) -> VideoInfo:
    """Read stream metadata with ffprobe."""
    path = str(path)
    if not Path(path).exists():
        raise FileNotFoundError(path)
    cmd = [
        ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
        "-show_streams", "-show_format", "-of", "json", path,
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise FFmpegError(f"ffprobe failed on {path}:\n{out.stderr.strip()}")
    data = json.loads(out.stdout)
    if not data.get("streams"):
        raise FFmpegError(f"no video stream found in {path}")
    st = data["streams"][0]
    fmt = data.get("format", {})

    rate = st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/0"
    try:
        fps = Fraction(rate)
    except (ZeroDivisionError, ValueError):
        fps = Fraction(0)
    if fps <= 0:
        try:
            fps = Fraction(st.get("r_frame_rate", "24/1"))
        except (ZeroDivisionError, ValueError):
            fps = Fraction(24)

    nb_frames, exact = 0, False
    for key in ("nb_frames", "nb_read_frames"):
        raw = st.get(key)
        if raw and str(raw).isdigit() and int(raw) > 0:
            nb_frames, exact = int(raw), True
            break
    if not nb_frames:
        duration = st.get("duration") or fmt.get("duration")
        if duration:
            nb_frames = int(round(float(duration) * float(fps)))

    bprs = st.get("bits_per_raw_sample")
    bprs = int(bprs) if bprs and str(bprs).isdigit() else None
    chroma, depth = _parse_pix_fmt(st.get("pix_fmt", ""), bprs)

    color_range = st.get("color_range") or ""
    if color_range not in ("tv", "pc"):
        # yuvj* formats are full range by definition; everything else defaults to limited.
        color_range = "pc" if str(st.get("pix_fmt", "")).startswith("yuvj") else "tv"

    return VideoInfo(
        path=path,
        width=int(st["width"]),
        height=int(st["height"]),
        pix_fmt=st.get("pix_fmt", ""),
        codec_name=st.get("codec_name", ""),
        fps=fps,
        nb_frames=nb_frames,
        frames_exact=exact,
        color_range=color_range,
        color_primaries=st.get("color_primaries"),
        color_transfer=st.get("color_transfer"),
        color_space=st.get("color_space"),
        bit_depth=depth,
        chroma=chroma,
    )


def _read_exact(pipe, n: int) -> bytes | None:
    """Read exactly n bytes from a pipe, or None at clean EOF."""
    chunks, remaining = [], n
    while remaining:
        chunk = pipe.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    if remaining == n:
        return None  # clean EOF on a frame boundary
    if remaining:
        raise FFmpegError(f"truncated frame: expected {n} bytes, got {n - remaining}")
    return b"".join(chunks)


def _seek_args(info: VideoInfo, seek_frame: int) -> list[str]:
    """Input-side seek. Modern ffmpeg decodes from the preceding keyframe to the exact
    timestamp, so this is frame-accurate *and* fast.

    ffmpeg keeps frames whose pts is >= the seek point, so we aim half a frame *early*:
    landing exactly on the target's pts risks float rounding dropping it and handing back
    its successor instead.
    """
    if seek_frame <= 0 or info.fps <= 0:
        return []
    return ["-ss", f"{max(0.0, (seek_frame - 0.5) / float(info.fps)):.6f}"]


def read_rgb(path: str | Path, start: int = 0, seek_frame: int = 0) -> Iterator[np.ndarray]:
    """Yield uint8 HxWx3 RGB frames. Colour conversion here is fine - it only feeds detection."""
    info = probe(path)
    cmd = [
        ffmpeg_exe(), "-nostdin", "-v", "error", *_seek_args(info, seek_frame),
        "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    nbytes = info.width * info.height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * 2)
    completed = False
    try:
        idx = 0
        while True:
            buf = _read_exact(proc.stdout, nbytes)
            if buf is None:
                completed = True
                break
            if idx >= start:
                yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width, 3)
            idx += 1
    finally:
        _shutdown(proc, completed)


@dataclass
class DepthFrame:
    """One decoded depth frame. ``y`` is always uint16 regardless of source bit depth."""

    y: np.ndarray
    u: np.ndarray | None
    v: np.ndarray | None


def read_depth(path: str | Path, info: VideoInfo | None = None,
               start: int = 0, seek_frame: int = 0) -> Iterator[DepthFrame]:
    """Yield DepthFrames with the luma plane bit-exact from the decoder.

    We request the source pixel format so swscale performs no conversion at all - crucially,
    no limited/full range rescale that would silently remap every depth code.
    """
    info = info or probe(path)
    cmd = [
        ffmpeg_exe(), "-nostdin", "-v", "error", *_seek_args(info, seek_frame),
        "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-f", "rawvideo",
        "-pix_fmt", info.decode_pix_fmt, "-",
    ]
    shapes = info.plane_shapes()
    dtype = np.uint8 if info.bit_depth == 8 else np.uint16
    nbytes = info.frame_nbytes
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * 2)
    completed = False
    try:
        idx = 0
        while True:
            buf = _read_exact(proc.stdout, nbytes)
            if buf is None:
                completed = True
                break
            if idx >= start:
                arr = np.frombuffer(buf, dtype)
                planes, off = [], 0
                for h, w in shapes:
                    planes.append(arr[off:off + h * w].reshape(h, w))
                    off += h * w
                y = planes[0].astype(np.uint16, copy=True)
                u = planes[1].copy() if len(planes) > 1 else None
                v = planes[2].copy() if len(planes) > 2 else None
                yield DepthFrame(y=y, u=u, v=v)
            idx += 1
    finally:
        _shutdown(proc, completed)


def _shutdown(proc: subprocess.Popen, completed: bool) -> None:
    """Tear down a reader process.

    *completed* says whether we drained the stream to EOF. If we stopped early - a preview
    grabbing frame 8 of 5000, say - ffmpeg dies writing into a closed pipe and reports a
    muxer error. That is us hanging up, not a decode failure, so it must not raise.
    """
    if not completed:
        proc.kill()
    if proc.stdout:
        proc.stdout.close()
    err = b""
    if proc.stderr:
        try:
            err = proc.stderr.read() or b""
        except Exception:
            pass
        proc.stderr.close()
    proc.wait()
    if completed and proc.returncode not in (0, None):
        text = err.decode("utf-8", "replace").strip()
        if text:
            raise FFmpegError(text)


class DepthWriter:
    """Encode DepthFrames back to a video, preserving fps, bit depth and colour tags."""

    def __init__(self, path: str | Path, info: VideoInfo, encoder: str = "libx265",
                 crf: int = 12, preset: str = "slow", lossless: bool = False):
        self.path = str(path)
        self.info = info
        self.chroma = "420" if info.chroma == "400" else info.chroma
        self.bit_depth = info.bit_depth
        self.dtype = np.uint8 if self.bit_depth == 8 else np.uint16
        self.neutral = 1 << (self.bit_depth - 1)
        self._synth_chroma = info.chroma == "400"

        in_pix = self._raw_pix_fmt()
        out_pix = f"yuv{self.chroma}p" if self.bit_depth == 8 \
            else f"yuv{self.chroma}p{self.bit_depth}le"

        fps = info.fps if info.fps > 0 else Fraction(24)
        cmd = [
            ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", in_pix,
            "-s", f"{info.width}x{info.height}", "-r", f"{fps.numerator}/{fps.denominator}",
            "-i", "-", "-an", "-sn", "-fps_mode", "passthrough",
            # Raw input carries no colour-range tag, so ffmpeg assumes limited. Without this
            # the encoder sees a tv->pc mismatch on a full-range depth map and swscale
            # quietly rescales every luma code. setrange only stamps metadata; it converts
            # nothing.
            "-vf", f"setrange=range={info.color_range}",
        ]
        cmd += self._codec_args(encoder, crf, preset, lossless, out_pix)
        cmd += self._color_args()
        cmd += [self.path]

        self.cmd = cmd
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._closed = False

    def _raw_pix_fmt(self) -> str:
        """Pixel format of the raw planes we feed on stdin (always has chroma)."""
        if self.bit_depth == 8:
            return f"yuv{self.chroma}p"
        return f"yuv{self.chroma}p{self.bit_depth}le"

    def _chroma_shape(self) -> tuple[int, int]:
        w, h = self.info.width, self.info.height
        if self.chroma == "420":
            return ((h + 1) // 2, (w + 1) // 2)
        if self.chroma == "422":
            return (h, (w + 1) // 2)
        return (h, w)

    def _codec_args(self, encoder, crf, preset, lossless, out_pix) -> list[str]:
        if lossless and encoder == "ffv1":
            return ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", out_pix]
        if encoder == "ffv1":
            return ["-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", out_pix]
        if encoder == "libx265":
            params = ["range=" + ("full" if self.info.color_range == "pc" else "limited")]
            if self.info.color_primaries and self.info.color_primaries != "unknown":
                params.append(f"colorprim={self.info.color_primaries}")
            if self.info.color_transfer and self.info.color_transfer != "unknown":
                params.append(f"transfer={self.info.color_transfer}")
            if self.info.color_space and self.info.color_space != "unknown":
                params.append(f"colormatrix={self.info.color_space}")
            if lossless:
                params.append("lossless=1")
            args = ["-c:v", "libx265", "-preset", preset, "-pix_fmt", out_pix,
                    "-x265-params", ":".join(params), "-tag:v", "hvc1"]
            if not lossless:
                args += ["-crf", str(crf)]
            return args
        if encoder == "libx264":
            args = ["-c:v", "libx264", "-preset", preset, "-pix_fmt", out_pix]
            args += ["-qp", "0"] if lossless else ["-crf", str(crf)]
            return args
        raise ValueError(f"unsupported encoder {encoder!r}")

    def _color_args(self) -> list[str]:
        args = ["-color_range", self.info.color_range]
        for flag, value in (
            ("-color_primaries", self.info.color_primaries),
            ("-color_trc", self.info.color_transfer),
            ("-colorspace", self.info.color_space),
        ):
            if value and value != "unknown":
                args += [flag, value]
        return args

    def write(self, frame: DepthFrame) -> None:
        if self._closed:
            raise FFmpegError("writer is closed")
        y = np.ascontiguousarray(frame.y.astype(self.dtype, copy=False))
        if y.shape != (self.info.height, self.info.width):
            raise ValueError(f"luma shape {y.shape} != {(self.info.height, self.info.width)}")
        planes = [y.tobytes()]
        ch, cw = self._chroma_shape()
        if self._synth_chroma or frame.u is None or frame.v is None:
            neutral = np.full((ch, cw), self.neutral, dtype=self.dtype)
            planes += [neutral.tobytes(), neutral.tobytes()]
        else:
            planes += [np.ascontiguousarray(frame.u.astype(self.dtype, copy=False)).tobytes(),
                       np.ascontiguousarray(frame.v.astype(self.dtype, copy=False)).tobytes()]
        try:
            self.proc.stdin.write(b"".join(planes))
        except BrokenPipeError as exc:
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            raise FFmpegError(f"encoder died:\n{err}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        err = self.proc.stderr.read() if self.proc.stderr else b""
        if self.proc.stderr:
            self.proc.stderr.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise FFmpegError(
                f"encoding failed (exit {self.proc.returncode}):\n"
                f"{err.decode('utf-8', 'replace').strip()}"
            )

    def __enter__(self) -> "DepthWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class GrayWriter:
    """Lossless 8-bit grayscale writer, used for the alpha mask cache."""

    def __init__(self, path: str | Path, width: int, height: int, fps: Fraction):
        self.path = str(path)
        suffix = Path(self.path).suffix.lower()
        codec = ["-c:v", "ffv1", "-level", "3", "-g", "1"] if suffix == ".mkv" \
            else ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast"]
        cmd = [
            ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{width}x{height}",
            "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
            "-an", "-fps_mode", "passthrough", *codec, "-pix_fmt", "gray", self.path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.shape = (height, width)
        self._closed = False

    def write(self, mask_u8: np.ndarray) -> None:
        arr = np.ascontiguousarray(mask_u8.astype(np.uint8, copy=False))
        if arr.shape != self.shape:
            raise ValueError(f"mask shape {arr.shape} != {self.shape}")
        try:
            self.proc.stdin.write(arr.tobytes())
        except BrokenPipeError as exc:
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            raise FFmpegError(f"mask encoder died:\n{err}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        err = self.proc.stderr.read() if self.proc.stderr else b""
        if self.proc.stderr:
            self.proc.stderr.close()
        self.proc.wait()
        if self.proc.returncode != 0:
            raise FFmpegError(f"mask encoding failed:\n{err.decode('utf-8', 'replace')}")

    def __enter__(self) -> "GrayWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_gray(path: str | Path) -> Iterator[np.ndarray]:
    """Yield uint8 HxW frames from a grayscale video (the mask cache)."""
    info = probe(path)
    cmd = [
        ffmpeg_exe(), "-nostdin", "-v", "error", "-i", str(path),
        "-map", "0:v:0", "-an", "-sn", "-dn",
        "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    nbytes = info.width * info.height
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * 2)
    completed = False
    try:
        while True:
            buf = _read_exact(proc.stdout, nbytes)
            if buf is None:
                completed = True
                break
            yield np.frombuffer(buf, np.uint8).reshape(info.height, info.width)
    finally:
        _shutdown(proc, completed)


def synth_test_video(path: str | Path, frames: Sequence[np.ndarray], fps: int = 24,
                     pix_fmt: str = "yuv420p10le", bit_depth: int = 10,
                     color_range: str = "tv", lossless: bool = True) -> None:
    """Write a test video from uint16 luma frames. Used by the test suite."""
    h, w = frames[0].shape
    out_pix = pix_fmt
    in_pix = f"yuv420p{bit_depth}le" if bit_depth > 8 else "yuv420p"
    dtype = np.uint16 if bit_depth > 8 else np.uint8
    neutral = 1 << (bit_depth - 1)
    codec = ["-c:v", "libx265", "-preset", "ultrafast",
             "-x265-params", f"lossless=1:range={'full' if color_range == 'pc' else 'limited'}"] \
        if lossless else ["-c:v", "libx265", "-crf", "12"]
    cmd = [
        ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", in_pix, "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-vf", f"setrange=range={color_range}",
        *codec, "-pix_fmt", out_pix, "-color_range", color_range,
        "-tag:v", "hvc1", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    ch, cw = (h + 1) // 2, (w + 1) // 2
    chroma = np.full((ch, cw), neutral, dtype=dtype).tobytes()
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f.astype(dtype)).tobytes())
        proc.stdin.write(chroma)
        proc.stdin.write(chroma)
    proc.stdin.close()
    err = proc.stderr.read()
    proc.stderr.close()
    proc.wait()
    if proc.returncode != 0:
        raise FFmpegError(err.decode("utf-8", "replace"))


def synth_rgb_video(path: str | Path, frames: Sequence[np.ndarray], fps: int = 24) -> None:
    """Write a test RGB video from uint8 HxWx3 frames (near-lossless)."""
    h, w = frames[0].shape[:2]
    cmd = [
        ffmpeg_exe(), "-nostdin", "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-qp", "0", "-preset", "veryfast",
        "-pix_fmt", "yuv444p", str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f.astype(np.uint8)).tobytes())
    proc.stdin.close()
    err = proc.stderr.read()
    proc.stderr.close()
    proc.wait()
    if proc.returncode != 0:
        raise FFmpegError(err.decode("utf-8", "replace"))
