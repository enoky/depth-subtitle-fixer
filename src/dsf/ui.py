"""Local Gradio app for tuning before committing to a full render.

The expensive step (detection) runs once per frame and is cached in the session, so the
brightness, dilate, feather and heal controls re-composite the displayed frame instantly.
The Render button then calls exactly the same code path as `dsf fix`.
"""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np

from .composite import composite_frame, resize_alpha
from .config import PipelineConfig, apply_profile, configure_model_cache
from .preview import contact_sheet
from .temporal import from_u8
from .videoio import probe


class Session:
    """Holds the loaded pair, cached per-frame masks and the reusable detector models."""

    def __init__(self) -> None:
        self.rgb_path: str | None = None
        self.depth_path: str | None = None
        self.rgb_info = None
        self.depth_info = None
        self.mask_cache: dict[int, np.ndarray] = {}
        self.frame_cache: dict[int, np.ndarray] = {}
        self.depth_frame_cache: dict[int, np.ndarray] = {}
        self.detectors = None
        self._detector_key: tuple | None = None

    def load(self, rgb_path: str, depth_path: str) -> str:
        from .pipeline import check_alignment

        self.rgb_path, self.depth_path = rgb_path, depth_path
        self.rgb_info, self.depth_info = probe(rgb_path), probe(depth_path)
        self.mask_cache.clear()
        self.frame_cache.clear()
        self.depth_frame_cache.clear()
        notes = check_alignment(self.rgb_info, self.depth_info)
        lines = [
            f"RGB   {self.rgb_info.width}x{self.rgb_info.height} @ "
            f"{float(self.rgb_info.fps):.3f} fps, {self.rgb_info.nb_frames} frames",
            f"depth {self.depth_info.width}x{self.depth_info.height} @ "
            f"{float(self.depth_info.fps):.3f} fps, {self.depth_info.nb_frames} frames, "
            f"{self.depth_info.bit_depth}-bit {self.depth_info.color_range}",
        ]
        lines += [f"note: {n}" for n in notes]
        return "\n".join(lines)

    def max_frame(self) -> int:
        counts = [n for n in (getattr(self.rgb_info, "nb_frames", 0),
                              getattr(self.depth_info, "nb_frames", 0)) if n]
        return max(0, (min(counts) if counts else 1) - 1)

    def get_detectors(self, cfg: PipelineConfig):
        from .detect import build_detectors

        key = (cfg.detect.detectors, cfg.detect.det_arch, cfg.detect.min_score,
               cfg.detect.batch_size, cfg.detect.device)
        if self.detectors is None or self._detector_key != key:
            self.detectors = build_detectors(cfg.detect.detectors, cfg.detect)
            self._detector_key = key
        return self.detectors

    def mask_for(self, index: int, cfg: PipelineConfig):
        """Returns ``(mask_u8, detections)``, cached per frame for instant re-compositing."""
        from .pipeline import masks_for_frames

        if index not in self.mask_cache:
            result = masks_for_frames(self.rgb_path, cfg, [index], self.rgb_info,
                                      detectors=self.get_detectors(cfg))
            self.mask_cache[index] = result.get(
                index, (np.zeros((self.rgb_info.height, self.rgb_info.width), np.uint8), []))
        return self.mask_cache[index]

    def rgb_for(self, index: int) -> np.ndarray:
        from .pipeline import sample_frames

        if index not in self.frame_cache:
            self.frame_cache.update(sample_frames(self.rgb_path, [index]))
        return self.frame_cache[index]

    def depth_for(self, index: int) -> np.ndarray:
        from .pipeline import sample_depth

        if index not in self.depth_frame_cache:
            frames = sample_depth(self.depth_path, [index], self.depth_info)
            for k, v in frames.items():
                self.depth_frame_cache[k] = v.y
        return self.depth_frame_cache[index]

    def invalidate_masks(self) -> None:
        self.mask_cache.clear()


def _config(profile, detectors, roi, scene_text, detect_every, min_persist,
            brightness, brightness_mode, dilate, feather, heal, heal_scope,
            heal_dilate, value_range) -> PipelineConfig:
    cfg = apply_profile(PipelineConfig(), profile)
    detect = dataclasses.replace(
        cfg.detect,
        detectors=tuple(detectors) if detectors else cfg.detect.detectors,
        detect_every=int(detect_every),
    )
    filters = dataclasses.replace(cfg.filters, roi=roi, scene_text=scene_text,
                                  min_persist_frames=int(min_persist))
    composite = dataclasses.replace(
        cfg.composite, brightness=float(brightness), brightness_mode=brightness_mode,
        dilate=int(dilate), feather=float(feather), heal=heal, heal_scope=heal_scope,
        heal_dilate=int(heal_dilate), value_range=value_range,
    )
    return dataclasses.replace(cfg, detect=detect, filters=filters, composite=composite)


def build_app(session: "Session | None" = None):
    """Construct the Gradio Blocks app. Split out from launch() so it can be built and
    inspected in tests without binding a port."""
    import gradio as gr

    configure_model_cache()
    session = session or Session()

    def on_load(rgb_file, depth_file):
        if not rgb_file or not depth_file:
            return "Select both an RGB clip and its depth map.", gr.update()
        info = session.load(rgb_file, depth_file)
        return info, gr.update(maximum=session.max_frame(), value=0)

    def render_frame(index, profile, detectors, roi, scene_text, detect_every, min_persist,
                     brightness, brightness_mode, dilate, feather, heal, heal_scope,
                     heal_dilate, value_range, recompute):
        if session.rgb_path is None:
            return None, "Load a clip pair first."
        cfg = _config(profile, detectors, roi, scene_text, detect_every, min_persist,
                      brightness, brightness_mode, dilate, feather, heal, heal_scope,
                      heal_dilate, value_range)
        index = int(index)
        if recompute:
            session.invalidate_masks()
        mask_u8, detections = session.mask_for(index, cfg)
        rgb = session.rgb_for(index)
        depth_y = session.depth_for(index)
        alpha_rgb = from_u8(mask_u8)
        alpha = resize_alpha(alpha_rgb, session.depth_info.width, session.depth_info.height)
        after = composite_frame(depth_y, alpha, cfg.composite,
                                session.depth_info.bit_depth, session.depth_info.color_range)
        sheet = contact_sheet(rgb, alpha_rgb, depth_y, after,
                              session.depth_info.bit_depth, detections=detections,
                              panel_width=560)
        covered = float((alpha_rgb > 0.02).mean()) * 100.0
        stats = (f"frame {index} | mask covers {covered:.2f}% of the frame | "
                 f"painted code "
                 f"{_painted_code(cfg, session.depth_info):.0f} of "
                 f"{(1 << session.depth_info.bit_depth) - 1}")
        return sheet[..., ::-1], stats  # BGR -> RGB for gradio

    def _painted_code(cfg, depth_info) -> float:
        from .composite import brightness_to_code, resolve_range

        return brightness_to_code(cfg.composite.brightness, depth_info.bit_depth,
                                  resolve_range(cfg.composite, depth_info.color_range))

    def render_clip(out_path, profile, detectors, roi, scene_text, detect_every, min_persist,
                    brightness, brightness_mode, dilate, feather, heal, heal_scope,
                    heal_dilate, value_range, max_frames, progress=gr.Progress()):
        if session.rgb_path is None:
            return "Load a clip pair first.", None
        from .pipeline import run_fix

        cfg = _config(profile, detectors, roi, scene_text, detect_every, min_persist,
                      brightness, brightness_mode, dilate, feather, heal, heal_scope,
                      heal_dilate, value_range)
        out = out_path.strip() if out_path else ""
        if not out:
            out = str(Path(tempfile.gettempdir()) /
                      (Path(session.depth_path).stem + "_fixed.mp4"))
        total = (int(max_frames) if max_frames else session.max_frame() + 1) or None

        def tick(n: int) -> None:
            if total:
                progress(min(1.0, n / total), desc=f"{n}/{total} frames")

        result = run_fix(session.rgb_path, session.depth_path, out, cfg,
                         max_frames=int(max_frames) if max_frames else None,
                         on_render=tick)
        return f"Wrote {result['frames']} frames to {out}", out

    with gr.Blocks(title="depth-subtitle-fixer") as demo:
        gr.Markdown(
            "## depth-subtitle-fixer\n"
            "Load an RGB clip and its DepthCrafter depth map, tune the mask and the painted "
            "grey level on a single frame, then render the whole clip."
        )
        with gr.Row():
            rgb_file = gr.Textbox(label="RGB clip (path)", placeholder=r"F:\clips\movie.mp4")
            depth_file = gr.Textbox(label="Depth map (path)",
                                    placeholder=r"F:\clips\movie_depth.mp4")
            load_btn = gr.Button("Load", variant="primary")
        info_box = gr.Textbox(label="Streams", lines=4, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Detection")
                profile = gr.Radio(["subtitles", "credits", "both"], value="subtitles",
                                   label="Profile")
                detectors = gr.CheckboxGroup(["doctr", "easyocr"], value=["doctr", "easyocr"],
                                             label="Detectors")
                roi = gr.Textbox(value="bottom:0.30", label="ROI",
                                 info="full | bottom:0.30 | top:0.20 | x0,y0,x1,y1")
                scene_text = gr.Radio(["keep", "mask"], value="keep",
                                      label="Text filmed in the scene")
                detect_every = gr.Slider(1, 8, value=2, step=1, label="Detect every N frames")
                min_persist = gr.Slider(1, 10, value=3, step=1, label="Min persistence frames")
                recompute = gr.Checkbox(value=False,
                                        label="Recompute mask (needed after changing "
                                              "detection settings)")

                gr.Markdown("### Depth repair")
                brightness = gr.Slider(0.0, 1.0, value=0.92, step=0.005, label="Brightness")
                brightness_mode = gr.Radio(["absolute", "relative"], value="absolute",
                                           label="Brightness mode")
                dilate = gr.Slider(0, 10, value=2, step=1, label="Dilate (px)")
                feather = gr.Slider(0.0, 8.0, value=1.5, step=0.1, label="Feather (sigma)")
                heal = gr.Radio(["edt", "none"], value="edt", label="Heal")
                heal_scope = gr.Radio(["glyph", "region"], value="glyph", label="Heal scope")
                heal_dilate = gr.Slider(0, 40, value=6, step=1, label="Heal halo (px)")
                value_range = gr.Radio(["auto", "tv", "pc"], value="auto",
                                       label="Luma code range")

            with gr.Column(scale=2):
                frame_idx = gr.Slider(0, 1, value=0, step=1, label="Frame")
                sheet = gr.Image(label="source | mask | depth before | depth after",
                                 type="numpy", height=620)
                stats = gr.Markdown()
                with gr.Row():
                    out_path = gr.Textbox(label="Output path",
                                          placeholder="leave blank for a temp file")
                    max_frames = gr.Number(label="Max frames (blank = all)", value=None)
                    render_btn = gr.Button("Render full clip", variant="primary")
                render_status = gr.Markdown()
                render_file = gr.File(label="Result")

        controls = [frame_idx, profile, detectors, roi, scene_text, detect_every, min_persist,
                    brightness, brightness_mode, dilate, feather, heal, heal_scope,
                    heal_dilate, value_range, recompute]

        load_btn.click(on_load, [rgb_file, depth_file], [info_box, frame_idx])
        for control in controls:
            control.change(render_frame, controls, [sheet, stats])

        render_btn.click(
            render_clip,
            [out_path, profile, detectors, roi, scene_text, detect_every, min_persist,
             brightness, brightness_mode, dilate, feather, heal, heal_scope, heal_dilate,
             value_range, max_frames],
            [render_status, render_file],
        )

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    build_app().launch(share=share, server_port=port, inbrowser=True)
