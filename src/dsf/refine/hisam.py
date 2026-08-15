"""Read the strokes with a model trained on strokes, instead of off a luma residual.

The residual estimates the picture without the writing and takes the difference, which
cannot work where the text is the brightness of what it sits on. Hi-SAM's SAM-TS is trained
on stroke masks and does not care what colour anything is. Measured on two credits, it
recovers 99.3% and 99.8% of the glyph pixels against the residual's 83.5% and 89.1%, and
where the residual collapses - a credit crossing hair at its own brightness - its worst
frame is 0.786 fgIoU against 0.226. See `docs/hisam-integration.md`.

It is not vendored. The model is about five thousand lines of third-party PyTorch, which is
the size of this whole project, and copying it in would double the code here with something
nobody here maintains. `scripts/fetch_hisam.py` clones it at a pinned commit into `models/`,
which is already where downloaded weights live and already ignored by git.

What this module owes the rest of the pipeline is a *shape*, and only that. Hi-SAM returns a
binary stroke mask with no notion of opacity, and a segmentation confidence is not an
opacity - a confidently-detected credit at 30% scores just as high as a solid one. The luma
model keeps measuring `level`; see `analyse_crop`.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np

from ..config import MODELS_DIR

#: Where `scripts/fetch_hisam.py` puts things.
CODE_DIR = MODELS_DIR / "hi_sam"
WEIGHTS_DIR = CODE_DIR / "pretrained_checkpoint"

#: The SAM backbone each head is built on, and the file the fetch script downloads for it.
_ENCODERS = {
    "vit_b": "sam_vit_b_01ec64.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_h": "sam_vit_h_4b8939.pth",
}

#: SAM-TS trained on TextSeg, which is design and poster text - the nearest published domain
#: to a title card. The HierText weights read document and scene text and score lower on
#: this material; the Total-Text ones lower again.
DEFAULT_HEAD = "sam_tss_l_textseg.pth"

_lock = threading.Lock()
_cached: tuple[tuple, object] | None = None


class Unavailable(RuntimeError):
    """Hi-SAM was asked for and is not installed. The message says how to fix it."""


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        raise Unavailable(
            f"{what} is missing: {path}\n"
            f"Run:  .venv/Scripts/python scripts/fetch_hisam.py\n"
            f"One of the two files it needs cannot be downloaded automatically and the "
            f"script will say so."
        )
    return path


def available(head: str = DEFAULT_HEAD, model_type: str = "vit_l") -> bool:
    """Whether the code and both checkpoints are present, without importing torch."""
    return ((CODE_DIR / "hi_sam" / "modeling" / "build.py").exists()
            and (WEIGHTS_DIR / head).exists()
            and (WEIGHTS_DIR / _ENCODERS[model_type]).exists())


def _build(head: str, model_type: str, device: str):
    """Construct the model with the weights merged in, without changing directory.

    The upstream builder resolves the SAM backbone through a path relative to the process's
    working directory, which is why its demo has to be run from its own folder. That is not
    something a library can do: the pipeline reads frames on a worker thread, and `chdir` is
    process-wide, so a load would move the ground under a decode running beside it. Building
    with no checkpoint and merging the two state dicts here by absolute path is the same
    arithmetic without the hazard.
    """
    import torch

    _require(CODE_DIR / "hi_sam" / "modeling" / "build.py", "the Hi-SAM source")
    head_path = _require(WEIGHTS_DIR / head, "the Hi-SAM head checkpoint")
    sam_path = _require(WEIGHTS_DIR / _ENCODERS[model_type], "the SAM backbone checkpoint")

    if str(CODE_DIR) not in sys.path:
        sys.path.insert(0, str(CODE_DIR))
    from hi_sam.modeling.build import model_registry  # noqa: E402
    from hi_sam.modeling.predictor import SamPredictor  # noqa: E402

    args = type("Args", (), {"checkpoint": None, "model_type": model_type,
                             "hier_det": False, "attn_layers": 1, "prompt_len": 12})()
    model = model_registry[model_type](args)

    state = torch.load(head_path, map_location="cpu")
    if any(k in state for k in ("optimizer", "lr_scheduler", "epoch")):
        state = state["model"]
    backbone = torch.load(sam_path, map_location="cpu")
    for key, value in backbone.items():
        state.setdefault(key, value)
    model.load_state_dict(state, strict=False)

    model.eval()
    model.to(device)
    return SamPredictor(model)


def predictor(head: str = DEFAULT_HEAD, model_type: str = "vit_l", device: str = "cuda"):
    """The loaded model, built once and kept.

    A ViT-L is 1.2 GB of weights and several seconds to construct, against 0.25 s to run a
    frame through it, so rebuilding per frame would be the whole cost of the feature.
    """
    global _cached
    key = (str(head), model_type, device)
    with _lock:
        if _cached is None or _cached[0] != key:
            _cached = (key, _build(head, model_type, device))
        return _cached[1]


def strokes(frame: np.ndarray, head: str = DEFAULT_HEAD, model_type: str = "vit_l",
            device: str = "cuda") -> np.ndarray:
    """The stroke mask for one uint8 RGB frame, as float32 in [0, 1] at the frame's size.

    Whole-frame rather than per detection crop. One forward pass costs the same at 1920x800
    as on an 888x448 crop - 0.25 s either way - because the model resizes its input to 1024
    internally and the extra pixels never reach it. So N crops would be N times the price of
    the one answer they are carved out of.
    """
    import torch

    pred = predictor(head, model_type, device)
    with torch.no_grad():
        pred.set_image(np.ascontiguousarray(frame))
        _, high_res, _, _ = pred.predict(multimask_output=False)
    mask = high_res[0] if high_res.ndim == 3 else high_res
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    return (np.asarray(mask) > 0).astype(np.float32)


def unload() -> None:
    """Drop the cached model. For tests, and for freeing 1.2 GB when switching heads."""
    global _cached
    with _lock:
        _cached = None
