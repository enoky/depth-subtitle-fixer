"""Native file-dialog helpers. The dialog itself needs a human, so what is tested here is
everything around it: the hints it is given, and that every way it can fail is survivable."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from dsf.filedialog import (
    VIDEO_FILETYPES, dialogs_available, pick_open, pick_path, pick_save, split_hint,
    suggest_output,
)


def test_split_hint_of_a_file_gives_its_folder_and_name(tmp_path):
    target = tmp_path / "movie.mp4"
    target.write_bytes(b"")
    folder, name = split_hint(str(target))
    assert folder == str(tmp_path)
    assert name == "movie.mp4"


def test_split_hint_of_a_folder_gives_the_folder(tmp_path):
    folder, name = split_hint(str(tmp_path))
    assert folder == str(tmp_path)
    assert name == ""


def test_split_hint_of_nothing_is_empty():
    assert split_hint("") == ("", "")
    assert split_hint("   ") == ("", "")


def test_split_hint_keeps_the_name_when_the_folder_does_not_exist(tmp_path):
    """Typing a path for a file that is not written yet must still seed the save dialog."""
    folder, name = split_hint(str(tmp_path / "nope" / "out.mp4"))
    assert folder == ""
    assert name == "out.mp4"


def test_suggest_output_sits_next_to_the_depth_map(tmp_path):
    depth = tmp_path / "movie_depth.mp4"
    out = Path(suggest_output(str(depth)))
    assert out.parent == tmp_path, "the result belongs beside its source"
    assert out.name == "movie_depth_fixed.mp4"
    assert Path(suggest_output(str(tmp_path / "m.mkv"))).suffix == ".mkv"


def test_suggest_output_of_nothing_is_nothing():
    assert suggest_output("") == ""


def test_video_filetypes_offer_an_escape_hatch():
    labels = [label for label, _ in VIDEO_FILETYPES]
    assert "All files" in labels, "users must be able to pick an unusual container"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        pick_path("sideways", "title")


def test_a_broken_interpreter_returns_empty_rather_than_raising():
    """A missing tkinter must degrade to 'keep what you typed', never to a traceback."""
    assert pick_path("open", "title", python="definitely-not-a-real-interpreter") == ""
    assert pick_open("title", python="definitely-not-a-real-interpreter") == ""
    assert pick_save("title", python="definitely-not-a-real-interpreter") == ""


def test_a_dialog_that_fails_returns_empty(tmp_path):
    """Simulated by pointing the helper at an interpreter that exits non-zero."""
    stub = tmp_path / "stub.py"
    stub.write_text("import sys; sys.exit(3)", encoding="utf-8")
    assert pick_path("open", "title", python=str(stub)) == ""


def test_availability_check_matches_this_interpreter():
    expected = True
    try:
        import tkinter  # noqa: F401
    except Exception:
        expected = False
    assert dialogs_available() is expected


def test_availability_check_is_false_for_a_missing_interpreter():
    assert dialogs_available(python="definitely-not-a-real-interpreter") is False


def test_dialog_never_runs_in_the_server_process():
    """Tk insists on owning a thread and Gradio serves each click from an arbitrary worker,
    so a dialog running in-process could wedge the whole server.

    Checked in a clean interpreter: other tests in this session import tkinter themselves,
    which would make an in-process check pass for the wrong reason.
    """
    probe = "import sys, dsf.filedialog; print('tkinter' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False", "dsf.filedialog imported tkinter into the server"


def test_suggest_output_for_a_frame_folder_is_a_folder(tmp_path):
    """Frames in, frames out - proposing an .mp4 for a sequence would be a dead end."""
    folder = tmp_path / "depth_png"
    folder.mkdir()
    assert suggest_output(str(folder)) == str(tmp_path / "depth_png_fixed")


def test_picking_a_frame_selects_its_folder(tmp_path):
    """A file dialog cannot select a folder, so picking any frame inside one stands in."""
    import cv2
    import numpy as np

    from dsf.filedialog import collapse_to_sequence

    folder = tmp_path / "rgb_png"
    folder.mkdir()
    for i in range(2):
        cv2.imwrite(str(folder / f"{i:08d}.png"), np.zeros((4, 4), np.uint8))
    assert collapse_to_sequence(str(folder / "00000000.png")) == str(folder)


def test_a_lone_file_is_left_alone(tmp_path):
    from dsf.filedialog import collapse_to_sequence

    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"")
    assert collapse_to_sequence(str(movie)) == str(movie)
    assert collapse_to_sequence("") == ""


def test_directory_mode_is_accepted():
    from dsf.filedialog import pick_directory

    assert pick_directory("t", python="definitely-not-a-real-interpreter") == ""
