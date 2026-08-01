"""Full pipeline against a synthetic clip pair. Needs model weights: run with --runslow."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import draw_subtitle_full, gradient_background, text_bbox
from dsf.cli import main
from dsf.config import PipelineConfig, apply_profile
from dsf.videoio import probe, read_depth, synth_rgb_video, synth_test_video

pytestmark = pytest.mark.slow

W, H, FRAMES = 640, 360, 16
TEXT = "HELLO WORLD"


def build_pair(tmp_path, depth_scale: float = 1.0):
    """An RGB clip with a burned-in subtitle, and a depth map corrupted where the text is."""
    bg = gradient_background(W, H)
    frame, fill, full = draw_subtitle_full(bg, TEXT, font_size=34, y_frac=0.85)
    rgb_frames = [frame.copy() for _ in range(FRAMES)]

    dw, dh = int(W * depth_scale), int(H * depth_scale)
    ramp = np.linspace(200, 700, dh, dtype=np.float32)[:, None].repeat(dw, axis=1)
    x0, y0, x1, y1 = text_bbox(full)
    sx, sy = dw / W, dh / H
    smear = (slice(max(0, int((y0 - 8) * sy)), int((y1 + 8) * sy)),
             slice(max(0, int((x0 - 8) * sx)), int((x1 + 8) * sx)))
    corrupted = ramp.copy()
    corrupted[smear] = 940.0  # the artefact DepthCrafter produces over text
    depth_frames = [np.clip(corrupted, 64, 940).astype(np.uint16) for _ in range(FRAMES)]

    rgb_path = tmp_path / "clip.mp4"
    depth_path = tmp_path / "depth.mp4"
    synth_rgb_video(rgb_path, rgb_frames, fps=24)
    synth_test_video(depth_path, depth_frames, fps=24, lossless=True)
    # `fill` is the glyph interior, the strictest thing the mask must cover; `full` adds the
    # outline, which the tool also masks by design.
    return rgb_path, depth_path, fill, full, smear


def subtitle_config() -> PipelineConfig:
    import dataclasses

    cfg = apply_profile(PipelineConfig(), "subtitles")
    # The synthetic clip is short and perfectly static; keep every frame in play.
    return dataclasses.replace(
        cfg,
        detect=dataclasses.replace(cfg.detect, detect_every=1, batch_size=4),
        filters=dataclasses.replace(cfg.filters, roi="bottom:0.35", min_persist_frames=2),
    )


def test_fix_paints_the_text_and_heals_the_smear(tmp_path):
    rgb_path, depth_path, fill, full, smear = build_pair(tmp_path)
    out = tmp_path / "fixed.mp4"

    from dsf.pipeline import run_fix

    cfg = subtitle_config()
    import dataclasses
    cfg = dataclasses.replace(cfg, composite=dataclasses.replace(
        cfg.composite, brightness=0.30, heal="edt", heal_dilate=10))

    result = run_fix(str(rgb_path), str(depth_path), str(out), cfg)
    assert result["frames"] == FRAMES

    before = [f.y for f in read_depth(depth_path)]
    after = [f.y for f in read_depth(out)]
    assert len(after) == FRAMES

    glyph = fill > 200
    mid = FRAMES // 2
    painted_code = 64 + 0.30 * (940 - 64)

    # The glyphs now sit near the requested level instead of the 940 artefact.
    glyph_after = after[mid][glyph].astype(np.float32)
    assert abs(float(np.median(glyph_after)) - painted_code) < 40, \
        f"glyphs landed at {np.median(glyph_after):.0f}, wanted ~{painted_code:.0f}"

    # The smeared halo is much closer to the surrounding depth than it was.
    halo = np.zeros((H, W), dtype=bool)
    halo[smear] = True
    halo &= ~glyph
    assert float(np.mean(after[mid][halo])) < float(np.mean(before[mid][halo])) - 100

    # Depth far from the text is left alone. This run uses the default CRF encode, so a
    # code or two of drift is the encoder, not the pipeline - bit-exactness is asserted in
    # test_clip_with_no_text_passes_through_unchanged, which encodes losslessly.
    top = (slice(0, 60), slice(0, W))
    np.testing.assert_allclose(after[mid][top].astype(np.int32),
                               before[mid][top].astype(np.int32), atol=2)


def test_brightness_controls_the_painted_level(tmp_path):
    import dataclasses

    from dsf.pipeline import run_fix

    rgb_path, depth_path, fill, full, _ = build_pair(tmp_path)
    glyph = fill > 200
    levels = {}
    for brightness in (0.20, 0.80):
        out = tmp_path / f"fixed_{brightness}.mp4"
        cfg = subtitle_config()
        cfg = dataclasses.replace(cfg, composite=dataclasses.replace(
            cfg.composite, brightness=brightness))
        run_fix(str(rgb_path), str(depth_path), str(out), cfg)
        after = [f.y for f in read_depth(out)]
        levels[brightness] = float(np.median(after[FRAMES // 2][glyph]))

    assert levels[0.80] > levels[0.20] + 300, \
        f"brightness had little effect: {levels}"


def test_depth_at_a_different_resolution_is_handled(tmp_path):
    """DepthCrafter routinely outputs smaller than the source clip."""
    from dsf.pipeline import run_fix

    rgb_path, depth_path, *_ = build_pair(tmp_path, depth_scale=0.5)
    out = tmp_path / "fixed.mp4"
    result = run_fix(str(rgb_path), str(depth_path), str(out), subtitle_config())

    assert result["frames"] == FRAMES
    src, dst = probe(depth_path), probe(out)
    assert (dst.width, dst.height) == (src.width, src.height)
    assert any("resolution differs" in n for n in result["notes"])


def test_detect_then_render_matches_a_single_pass(tmp_path):
    """The mask cache exists so brightness can be re-tuned without re-detecting."""
    rgb_path, depth_path, fill, full, _ = build_pair(tmp_path)
    mask_path = tmp_path / "masks.mkv"
    one_pass = tmp_path / "one_pass.mp4"
    two_pass = tmp_path / "two_pass.mp4"

    assert main(["fix", "--rgb", str(rgb_path), "--depth", str(depth_path),
                 "--out", str(one_pass), "--roi", "bottom:0.35", "--detect-every", "1",
                 "--min-persist-frames", "2", "--brightness", "0.4"]) == 0
    assert main(["detect", "--rgb", str(rgb_path), "--out-mask", str(mask_path),
                 "--roi", "bottom:0.35", "--detect-every", "1",
                 "--min-persist-frames", "2"]) == 0
    assert main(["render", "--depth", str(depth_path), "--mask", str(mask_path),
                 "--out", str(two_pass), "--brightness", "0.4"]) == 0

    a = [f.y for f in read_depth(one_pass)]
    b = [f.y for f in read_depth(two_pass)]
    assert len(a) == len(b) == FRAMES
    glyph = fill > 200
    assert abs(float(np.median(a[8][glyph])) - float(np.median(b[8][glyph]))) < 10

    from dsf.maskcache import cache_matches, load_meta

    assert cache_matches(mask_path, rgb_path)
    meta = load_meta(mask_path)
    assert meta is not None and meta.frames == FRAMES


def test_preview_writes_contact_sheets(tmp_path):
    import cv2

    rgb_path, depth_path, *_ = build_pair(tmp_path)
    out_dir = tmp_path / "previews"
    assert main(["preview", "--rgb", str(rgb_path), "--depth", str(depth_path),
                 "--frames", "2,8", "--out-dir", str(out_dir),
                 "--roi", "bottom:0.35", "--detect-every", "1"]) == 0

    sheets = sorted(out_dir.glob("*.png"))
    assert len(sheets) == 2
    img = cv2.imread(str(sheets[0]))
    assert img is not None and img.shape[0] > 0


def test_clip_with_no_text_passes_through_unchanged(tmp_path):
    """No detections must mean no edits - not a silently degraded depth map."""
    from dsf.pipeline import run_fix

    blank = [gradient_background(W, H) for _ in range(8)]
    depth = [np.full((H, W), 500, dtype=np.uint16) for _ in range(8)]
    rgb_path, depth_path = tmp_path / "blank.mp4", tmp_path / "blankd.mp4"
    synth_rgb_video(rgb_path, blank, fps=24)
    synth_test_video(depth_path, depth, fps=24, lossless=True)

    out = tmp_path / "out.mp4"
    import dataclasses
    cfg = subtitle_config()
    cfg = dataclasses.replace(cfg, encode=dataclasses.replace(cfg.encode, lossless=True))
    run_fix(str(rgb_path), str(depth_path), str(out), cfg)

    for frame in read_depth(out):
        np.testing.assert_array_equal(frame.y, np.full((H, W), 500, dtype=np.uint16))
