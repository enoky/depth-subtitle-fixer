"""The Gradio app must at least construct - a broken control wiring is otherwise only
discovered when a user launches it."""

from __future__ import annotations

import pytest

gr = pytest.importorskip("gradio")


def test_app_builds_without_binding_a_port():
    from dsf.ui import Session, build_app

    demo = build_app(Session())
    assert demo is not None
    assert demo.title == "depth-subtitle-fixer"


def test_session_reports_nothing_loaded():
    from dsf.ui import Session

    session = Session()
    assert session.rgb_path is None
    assert session.max_frame() == 0
