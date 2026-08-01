"""Native OS file dialogs for the local Gradio app.

Gradio's own file input uploads whatever you pick into its temp cache. That is the right
behaviour for a photo and completely wrong for a two-hour 4K master - the pipeline wants a
*path* to open and stream, not a multi-gigabyte copy. So the Browse buttons open a real OS
dialog and hand back the path the user chose.

The dialog runs in a short-lived subprocess rather than in-process. Tk insists on owning a
thread, Gradio serves each click from an arbitrary worker, and a wedged Tk loop would take
the whole server with it. A subprocess cannot do that, costs a fraction of a second, and
exits cleanly however the user dismisses the window.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

#: (label, space-separated patterns) pairs, in the order the dialog should offer them.
VIDEO_FILETYPES: list[tuple[str, str]] = [
    ("Video and frames",
     "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpg *.mpeg *.ts *.m2ts *.y4m "
     "*.png *.tif *.tiff"),
    ("Video files", "*.mp4 *.mkv *.mov *.avi *.webm *.m4v *.mpg *.mpeg *.ts *.m2ts *.y4m"),
    ("Image sequences", "*.png *.tif *.tiff"),
    ("All files", "*.*"),
]

_SCRIPT = r"""
import json, sys
import tkinter as tk
from tkinter import filedialog

opts = json.loads(sys.argv[1])
root = tk.Tk()
root.withdraw()
# Without this the dialog can open behind the browser window that spawned it.
root.attributes("-topmost", True)
kwargs = {
    "title": opts["title"],
    "filetypes": [tuple(ft) for ft in opts["filetypes"]],
}
if opts.get("initialdir"):
    kwargs["initialdir"] = opts["initialdir"]
if opts["mode"] == "directory":
    chosen = filedialog.askdirectory(title=opts["title"],
                                     initialdir=opts.get("initialdir") or None)
elif opts["mode"] == "save":
    if opts.get("initialfile"):
        kwargs["initialfile"] = opts["initialfile"]
    kwargs["defaultextension"] = opts.get("defaultextension", "")
    chosen = filedialog.asksaveasfilename(**kwargs)
else:
    chosen = filedialog.askopenfilename(**kwargs)
root.destroy()
sys.stdout.write(chosen or "")
"""


def dialogs_available(python: str | None = None) -> bool:
    """Whether this interpreter can actually open a dialog (tkinter is often absent)."""
    try:
        proc = subprocess.run([python or sys.executable, "-c", "import tkinter"],
                              capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


def split_hint(current: str) -> tuple[str, str]:
    """Turn whatever is already in the box into (starting directory, starting filename)."""
    if not current or not current.strip():
        return "", ""
    path = Path(current.strip()).expanduser()
    if path.is_dir():
        return str(path), ""
    parent = path.parent
    return (str(parent) if str(parent) not in ("", ".") and parent.exists() else "",
            path.name)


def pick_path(mode: str, title: str, current: str = "",
              filetypes: list[tuple[str, str]] | None = None,
              default_extension: str = "", python: str | None = None,
              timeout: float = 600.0) -> str:
    """Open a dialog and return the chosen path, or "" if it was cancelled or unavailable.

    Never raises: a missing tkinter, a killed dialog or a user pressing Escape all mean the
    same thing to the caller - keep whatever path was already there.
    """
    if mode not in ("open", "save", "directory"):
        raise ValueError(f"mode must be 'open', 'save' or 'directory'; got {mode!r}")
    initialdir, initialfile = split_hint(current)
    opts = {
        "mode": mode,
        "title": title,
        "filetypes": filetypes if filetypes is not None else VIDEO_FILETYPES,
        "initialdir": initialdir,
        "initialfile": initialfile,
        "defaultextension": default_extension,
    }
    try:
        proc = subprocess.run(
            [python or sys.executable, "-c", _SCRIPT, json.dumps(opts)],
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout.strip()


def pick_open(title: str, current: str = "", **kwargs) -> str:
    return pick_path("open", title, current, **kwargs)


def pick_save(title: str, current: str = "", **kwargs) -> str:
    return pick_path("save", title, current, default_extension=".mp4", **kwargs)


def pick_directory(title: str, current: str = "", **kwargs) -> str:
    """Choose a folder - an image sequence in, or a folder of frames out."""
    return pick_path("directory", title, current, **kwargs)


def collapse_to_sequence(path: str) -> str:
    """If a still inside a frame folder was picked, use the folder.

    Saves needing a separate button for sequences: a file dialog cannot select a folder, but
    picking any frame inside one says just as clearly which folder was meant.
    """
    if not path:
        return path
    from . import sequence

    candidate = Path(path)
    if (candidate.is_file()
            and candidate.suffix.lower() in sequence.IMAGE_SUFFIXES
            and sequence.is_sequence(candidate.parent)):
        return str(candidate.parent)
    return path


def suggest_output(depth_path: str, suffix: str = "_fixed") -> str:
    """Default output beside the depth map, so the box is rarely empty.

    A folder of frames suggests a sibling folder, not a video file - the frames have to come
    back out as frames.
    """
    if not depth_path:
        return ""
    path = Path(depth_path)
    if path.is_dir():
        return str(path.with_name(f"{path.name}{suffix}"))
    return str(path.with_name(f"{path.stem}{suffix}{path.suffix or '.mp4'}"))
