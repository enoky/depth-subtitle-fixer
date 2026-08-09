"""Run the per-frame full-frame stages on the GPU when there is one.

Detection was on the GPU from the start; the mask chain behind it was not. Composing patches
into a full-frame alpha, converting it to bytes and back, the temporal median, and the prior's
threshold-and-or over a twenty-one frame window all touch every pixel of every frame, and on a
1920x1080 clip they measured 33 of the 99 ms a scanned frame cost - while the card that had
just run the detector sat idle.

Nothing here defines what the pipeline computes. Every operation has a numpy original behind
it, `CpuOps` *is* that original rather than a second copy of it, and `tests/test_accel.py`
holds the two against each other on real-shaped data. The GPU path is only ever an
accelerator, so a machine without one - or with a CPU-only OpenCV - loses speed and nothing
else.

Buffers travelling between these operations are opaque: ndarrays under `CpuOps`,
`cv2.cuda.GpuMat` under `CudaOps`. Callers move a frame through the chain without knowing
which, and `download` at the end returns an ndarray. Keeping the chain GPU-resident is the
point - converting at each step would spend more on the bus than the kernels save.

Every operation here measured bit-identical to its numpy original, including `convertTo`,
because OpenCV's `cvRound` and `np.rint` both round halves to even. The one case that cannot
be made to match - the even-length median - says so at `CudaOps.median` and falls back to the
CPU rather than being quietly approximated.
"""

from __future__ import annotations

import functools
import os
import threading

import cv2
import numpy as np

from .refine.strokes import compose_alpha, compose_levels
from .temporal import _median_u8, _sorting_network, to_u8

#: Set to "cpu" or "cuda" to override the resolved backend. Exists so the same scan can be run
#: both ways without editing anything, which is how the parity and benchmark numbers are taken.
ENV_OVERRIDE = "DSF_ACCEL"


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Whether this OpenCV build can see a CUDA device.

    A CPU-only OpenCV answers False rather than raising: `cv2.cuda` is a namespace in every
    build, it is only ever empty of devices.
    """
    try:
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:  # noqa: BLE001 - some builds omit the namespace entirely
        return False


@functools.lru_cache(maxsize=1)
def _warn_no_cuda_opencv() -> None:
    """Say so, once, when torch has a GPU and OpenCV does not.

    This is the shape a broken install takes: `pip install -e .` pulls `opencv-python` back in
    as a dependency of docTR or easyocr and unpacks a CPU-only build over the CUDA one. The
    scan then silently gets slower, which is exactly the failure worth being loud about - the
    whole point of the custom build is that it is not the stock wheel.
    """
    import warnings

    warnings.warn(
        "OpenCV reports no CUDA device, so the mask chain is running on the CPU. If this "
        "machine has an NVIDIA GPU the CUDA-enabled OpenCV build has probably been "
        "overwritten by a plain `pip install` - re-run scripts/install_opencv_cuda.ps1.",
        RuntimeWarning, stacklevel=3,
    )


def resolve(device: str = "auto") -> str:
    """Which backend to use: ``"cuda"`` or ``"cpu"``.

    Takes the same `--device` string the detectors take, so one flag drives both and a run
    forced onto the CPU does not quietly keep half of itself on the card.
    """
    override = os.environ.get(ENV_OVERRIDE, "").strip().lower()
    if override in ("cpu", "cuda"):
        device = override
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        return "cuda" if cuda_available() else "cpu"

    from .detect.base import resolve_device

    if resolve_device("auto") != "cuda":
        return "cpu"  # no GPU at all; nothing to warn about
    if not cuda_available():
        _warn_no_cuda_opencv()
        return "cpu"
    return "cuda"


class CpuOps:
    """The numpy original. Buffers are plain ndarrays, and every method here is a one-liner
    over code that already existed - this is the reference, not a second implementation."""

    name = "cpu"

    def reserve(self, rings: dict[str, int]) -> None:
        pass  # numpy has no pool to size

    # ---------------------------------------------------------------- composition
    def compose(self, patches, height: int, width: int, normalised: bool) -> np.ndarray:
        return compose_alpha(patches, height, width, normalised=normalised)

    def compose_levels(self, patches, height: int, width: int) -> np.ndarray:
        return compose_levels(patches, height, width)

    def to_u8(self, alpha: np.ndarray, slot: str = "shape") -> np.ndarray:
        return to_u8(alpha)  # *slot* only means something to the pooled backend

    # ---------------------------------------------------------------- reductions
    def peak(self, buf_u8: np.ndarray) -> float:
        return float(buf_u8.max())

    def any(self, buf_u8: np.ndarray) -> bool:
        return bool(buf_u8.any())

    # ---------------------------------------------------------------- elementwise
    def over_127(self, buf_u8: np.ndarray) -> np.ndarray:
        """Where the mask reads as a stroke, as 0/255 rather than a bool.

        0/255 so the result can be dilated and then used to select with a bitwise and, which
        is what makes the GPU path a single kernel; on this side it is only a dtype. Callers
        that want a bool ask for ``> 0``.
        """
        return np.where(buf_u8 > 127, np.uint8(255), np.uint8(0))

    def or_into(self, acc: np.ndarray | None, buf_u8: np.ndarray) -> np.ndarray:
        if acc is None:
            return buf_u8.copy()
        np.bitwise_or(acc, buf_u8, out=acc)
        return acc

    def select(self, buf_u8: np.ndarray, keep_u8: np.ndarray) -> np.ndarray:
        """*buf_u8* where *keep_u8* is set, zero elsewhere. *keep_u8* is 0/255."""
        return np.bitwise_and(buf_u8, keep_u8)

    def dilate(self, buf_u8: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        return cv2.dilate(buf_u8, kernel)

    def scale_by(self, shape_u8: np.ndarray, level_u8: np.ndarray) -> np.ndarray:
        """The stroke shape scaled down by the opacity its text is showing at.

        ``to_u8(from_u8(shape) * from_u8(level))``, which is what this replaced.
        """
        return to_u8((shape_u8.astype(np.float32) / 255.0)
                     * (level_u8.astype(np.float32) / 255.0))

    def median(self, window) -> np.ndarray:
        return _median_u8(list(window))

    def max_reduce(self, window) -> np.ndarray:
        return np.maximum.reduce(list(window))

    # ---------------------------------------------------------------- transfer
    def upload(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def download(self, buf: np.ndarray) -> np.ndarray:
        return buf

    def sync(self) -> None:
        pass


#: Ring depth for buffers consumed before the next one of their kind is asked for. Two rather
#: than one so a result can still be read while its successor is being written.
_TRANSIENT = 3


class CudaOps:
    """The same operations on `cv2.cuda.GpuMat`, on this thread's own stream.

    One instance per thread, never shared. The scanner runs three clip workers by default and
    `cv2.cuda` filter objects carry internal state, so a shared instance would have two
    workers writing into each other's frames. This mirrors the way `SharedDetector` in
    `scripts/scan_for_text.py` keeps docTR's mutable preprocessor per-thread.

    **Buffers are pooled, and that is the whole reason this is faster.** Measured at 1080p, a
    `cv2.cuda.min` that allocates its own destination costs 0.335 ms while the same call into
    a preallocated one costs 0.0068 ms - fifty times less. The kernels were never the
    expense; `cudaMalloc` was. Letting every operation allocate would make the prior's
    twenty-one thresholds cost 7 ms a frame instead of 0.14 ms, and the port would be pointless.

    So each purpose draws from a ring of buffers and cycles through it. A ring must be longer
    than the pipeline can hold onto that purpose's output, or a buffer still sitting in the
    prior's window would be handed out again and overwritten mid-flight. Those depths are not
    guessable from here - they follow the caller's window settings - so the caller names its
    rings through `reserve` and everything unnamed is assumed transient.
    """

    name = "cuda"

    def __init__(self):
        self._stream = cv2.cuda.Stream()
        self._rings: dict[tuple, list] = {}
        self._next: dict[tuple, int] = {}
        self._filters: dict[tuple, object] = {}
        self._depth: dict[str, int] = {}
        self._shape: tuple[int, int] | None = None

    def reserve(self, rings: dict[str, int]) -> None:
        """Declare how many frames each named ring must outlast.

        Grows rings and never shrinks them, so a long clip following a short one cannot end up
        working from a ring sized for the short one.
        """
        for key, depth in rings.items():
            self._depth[key] = max(self._depth.get(key, _TRANSIENT), int(depth))

    # ---------------------------------------------------------------- pooling
    def _lease(self, key: str, height: int, width: int, kind: int, depth: int | None = None):
        if depth is None:
            depth = self._depth.get(key, _TRANSIENT)
        if self._shape is not None and (height, width) != self._shape:
            # A clip of a different size. Rings are keyed by shape, so without this a scan
            # over a folder of mixed resolutions would keep one full set of buffers per size
            # it had ever seen - a few hundred megabytes of VRAM per worker, none of it
            # reachable again. Safe here because a worker finishes one clip before it starts
            # the next, so nothing still in flight is the old size.
            self._rings.clear()
            self._next.clear()
        self._shape = (height, width)
        slot = (key, height, width, kind)
        ring = self._rings.get(slot)
        if ring is None:
            ring = []
            self._rings[slot] = ring
            self._next[slot] = 0
        if len(ring) < depth:
            # Grown lazily: a clip shorter than the window never pays for the whole ring.
            ring.append(cv2.cuda.GpuMat(height, width, kind))
        index = self._next[slot] % len(ring)
        self._next[slot] = index + 1
        return ring[index]

    def _like(self, src, key: str, kind: int, depth: int | None = None):
        width, height = src.size()
        return self._lease(key, height, width, kind, depth)

    def _dilate_filter(self, kernel: np.ndarray):
        slot = (kernel.shape, kernel.tobytes())
        found = self._filters.get(slot)
        if found is None:
            found = cv2.cuda.createMorphologyFilter(cv2.MORPH_DILATE, cv2.CV_8UC1, kernel)
            self._filters[slot] = found
        return found

    # ---------------------------------------------------------------- composition
    def _compose_into(self, patches, height, width, key, source_of):
        """Max each patch into its own rectangle of a zeroed full-frame buffer.

        The per-patch array is still built with numpy: patches are crops a few hundred pixels
        across, so that work is trivial, and doing it on the card would cost several kernel
        launches per detection to save nothing.
        """
        out = self._lease(key, height, width, cv2.CV_32FC1, 1)
        out.setTo(0.0, stream=self._stream)
        for patch in patches:
            ph, pw = patch.shape
            y0, x0 = patch.y0, patch.x0
            y1, x1 = min(height, y0 + ph), min(width, x0 + pw)
            if y1 <= y0 or x1 <= x0:
                continue
            source = np.ascontiguousarray(source_of(patch)[: y1 - y0, : x1 - x0])
            gpu = cv2.cuda.GpuMat()
            gpu.upload(source, stream=self._stream)
            roi = cv2.cuda.GpuMat(out, (x0, y0, x1 - x0, y1 - y0))
            cv2.cuda.max(roi, gpu, roi, stream=self._stream)
        return out

    def compose(self, patches, height: int, width: int, normalised: bool):
        pick = (lambda p: p.normalised) if normalised else (lambda p: p.alpha)
        return self._compose_into(patches, height, width, "compose_shape", pick)

    def compose_levels(self, patches, height: int, width: int):
        def source_of(patch):
            # Identical to `strokes.compose_levels`: the patch's level wherever its normalised
            # shape covers something, nothing elsewhere.
            return np.where(patch.normalised > 0.25, np.float32(patch.level), np.float32(0.0))

        return self._compose_into(patches, height, width, "compose_levels", source_of)

    def to_u8(self, alpha, slot: str = "shape"):
        """Bit-identical to `temporal.to_u8`: `cvRound` and `np.rint` agree on halves.

        *slot* names the ring to draw from. The caller passes different names for results with
        different lifetimes - a stroke shape is let go within a few frames while the level map
        beside it is held for the whole prior window, and sharing one ring would size both for
        the longer of the two.
        """
        out = self._like(alpha, slot, cv2.CV_8UC1)
        alpha.convertTo(rtype=cv2.CV_8U, dst=out, alpha=255.0, beta=0.0, stream=self._stream)
        return out

    # ---------------------------------------------------------------- reductions
    def peak(self, buf_u8) -> float:
        """The one operation here the GPU is worse at, and it is kept anyway.

        Measured at 1080p: 0.54 ms against numpy's 0.016 ms, because collapsing a frame to a
        single number means waiting for the stream and then a reduction that cannot overlap
        with anything. It stays because the alternative is worse, not because it is good -
        fetching the frame to answer it on the CPU costs 0.31 ms of transfer *plus* the 0.3 ms
        numpy then spends on the max, and leaves the buffer needlessly on the host.

        Roughly half a millisecond of a 64 ms frame, so it has not been worth restructuring
        the prior to avoid. If it ever is, the peak of a level map is knowable without looking
        at the frame at all - it is the largest level among the patches that composed it.
        """
        self._stream.waitForCompletion()
        return float(cv2.cuda.minMax(buf_u8)[1])

    def any(self, buf_u8) -> bool:
        self._stream.waitForCompletion()
        return cv2.cuda.countNonZero(buf_u8) > 0

    # ---------------------------------------------------------------- elementwise
    def over_127(self, buf_u8):
        out = self._like(buf_u8, "thresh", cv2.CV_8UC1)
        cv2.cuda.threshold(buf_u8, 127, 255, cv2.THRESH_BINARY, dst=out, stream=self._stream)
        return out

    def or_into(self, acc, buf_u8):
        if acc is None:
            acc = self._like(buf_u8, "trusted", cv2.CV_8UC1)
            buf_u8.copyTo(stream=self._stream, dst=acc)
            return acc
        return cv2.cuda.bitwise_or(acc, buf_u8, acc, stream=self._stream)

    def select(self, buf_u8, keep_u8):
        out = self._like(buf_u8, "select", cv2.CV_8UC1)
        cv2.cuda.bitwise_and(buf_u8, keep_u8, out, stream=self._stream)
        return out

    def dilate(self, buf_u8, kernel: np.ndarray):
        out = self._like(buf_u8, "dilate", cv2.CV_8UC1)
        self._dilate_filter(kernel).apply(buf_u8, dst=out, stream=self._stream)
        return out

    def scale_by(self, shape_u8, level_u8):
        """As `CpuOps.scale_by`, as one scaled multiply. Measured bit-identical."""
        out = self._like(shape_u8, "scaled", cv2.CV_8UC1)
        cv2.cuda.multiply(shape_u8, level_u8, dst=out, scale=1.0 / 255.0, dtype=cv2.CV_8U,
                          stream=self._stream)
        return out

    def median(self, window):
        """The compare-exchange network `temporal._median_u8` runs, as GPU min/max.

        Integer min and max in the same order, so the odd case is exact rather than close.

        The even case is not, and is not attempted: the numpy original floors the average of
        the middle pair, and no CUDA arithmetic op floors - `addWeighted` rounds, and rounds
        halves to even, which disagrees with a floor on exactly the values that matter. An
        even window only arises where one is truncated at a clip boundary, so at most twice
        per clip, and it is answered on the CPU where the answer is right.
        """
        items = list(window)
        count = len(items)
        if count % 2 == 0:
            centre = _median_u8([self.download(item) for item in items])
            return self.upload(centre)
        if count == 3:
            # The median of three needs no sort: the larger of the two smaller ones.
            a, b, c = items
            lo = self._like(a, "med_lo", cv2.CV_8UC1)
            hi = self._like(a, "med_hi", cv2.CV_8UC1)
            cv2.cuda.min(a, b, lo, stream=self._stream)
            cv2.cuda.max(a, b, hi, stream=self._stream)
            cv2.cuda.min(hi, c, hi, stream=self._stream)
            out = self._like(a, "median", cv2.CV_8UC1)
            cv2.cuda.max(lo, hi, out, stream=self._stream)
            return out
        # Longer odd windows: sort in scratch, then copy the middle into a retained buffer.
        work = [self._like(items[0], f"sort{i}", cv2.CV_8UC1) for i in range(count)]
        for slot, frame in zip(work, items):
            frame.copyTo(stream=self._stream, dst=slot)
        low = self._like(items[0], "med_lo", cv2.CV_8UC1)
        for lo, hi in _sorting_network(count):
            # `low` holds min(lo, hi) before `hi` is overwritten, exactly as the numpy
            # original does; the buffers are then swapped so the one just freed becomes the
            # next comparison's scratch, and no ring is drawn from inside the loop.
            cv2.cuda.min(work[lo], work[hi], low, stream=self._stream)
            cv2.cuda.max(work[lo], work[hi], work[hi], stream=self._stream)
            low, work[lo] = work[lo], low
        out = self._like(items[0], "median", cv2.CV_8UC1)
        work[count // 2].copyTo(stream=self._stream, dst=out)
        return out

    def max_reduce(self, window):
        items = list(window)
        out = self._like(items[0], "median", cv2.CV_8UC1)
        items[0].copyTo(stream=self._stream, dst=out)
        for frame in items[1:]:
            cv2.cuda.max(out, frame, out, stream=self._stream)
        return out

    # ---------------------------------------------------------------- transfer
    def upload(self, arr: np.ndarray):
        gpu = cv2.cuda.GpuMat()
        gpu.upload(np.ascontiguousarray(arr), stream=self._stream)
        return gpu

    def download(self, buf) -> np.ndarray:
        self._stream.waitForCompletion()
        return buf.download()

    def sync(self) -> None:
        self._stream.waitForCompletion()


_LOCAL = threading.local()


def ops(device: str = "auto", rings: dict[str, int] | None = None):
    """This thread's operations for *device*, with *rings* deep enough for the caller's windows.

    Per-thread because the CUDA side owns a stream, a set of filters and a pool of buffers,
    and two clip workers sharing those would overwrite each other's frames.
    """
    backend = resolve(device)
    cached = getattr(_LOCAL, backend, None)
    if cached is None:
        cached = CudaOps() if backend == "cuda" else CpuOps()
        setattr(_LOCAL, backend, cached)
    if rings:
        cached.reserve(rings)
    return cached
