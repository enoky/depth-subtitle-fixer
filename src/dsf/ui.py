"""Local Gradio app for tuning before committing to a full render.

The expensive step (detection) runs once per frame and is cached in the session, so the
brightness, dilate, feather and heal controls re-composite the displayed frame instantly.
The Render button then calls exactly the same code path as `dsf fix`.

Every control is generated from the table in `controls.py`, which the test suite compares
against the CLI's argument map - so a knob cannot be reachable from the command line and
quietly missing here.
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import numpy as np

from .composite import composite_frame, resize_alpha
from .config import PipelineConfig, configure_model_cache
from .controls import (CONFIG_KNOBS, GROUP_TITLES, KNOBS, build_config, defaults,
                       group_knobs, make_component, profile_defaults)
from .filedialog import (collapse_to_sequence, dialogs_available, pick_directory,
                         pick_open, pick_save, suggest_output)
from .media import probe
from .preview import contact_sheet
from .temporal import from_u8

#: A Gradio Button is a bare flex child of its Row, while a labelled Textbox is wrapped in a
#: padded block. Left alone the button therefore floats against the label rather than the
#: field. Pinning it to the bottom and lifting it by the block's own 10px padding lands it
#: level with the input it belongs to. equal_height on the Row is not the answer - that
#: stretches the button to the full height of the tallest thing in the row.
_CSS = """
.dsf-browse { align-self: flex-end; margin-bottom: 10px; flex-grow: 0; }
"""


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
        kind = "frames" if self.rgb_info.codec_name == "imageseq" else "video"
        lines = [
            f"RGB   {self.rgb_info.width}x{self.rgb_info.height} @ "
            f"{float(self.rgb_info.fps):.3f} fps, {self.rgb_info.nb_frames} {kind}",
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

    def has_mask(self, index: int) -> bool:
        return index in self.mask_cache

    def mask_for(self, index: int, cfg: PipelineConfig, progress=None):
        """Returns ``(mask_u8, detections)``, cached per frame for instant re-compositing."""
        from .pipeline import masks_for_frames

        if index not in self.mask_cache:
            result = masks_for_frames(self.rgb_path, cfg, [index], self.rgb_info,
                                      detectors=self.get_detectors(cfg),
                                      progress=progress)
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
                self.depth_frame_cache[k] = v.plane
        return self.depth_frame_cache[index]

    def invalidate_masks(self) -> None:
        self.mask_cache.clear()


def build_app(session: "Session | None" = None, native_dialogs: bool = True):
    """Construct the Gradio Blocks app. Split out from launch() so it can be built and
    inspected in tests without binding a port.

    *native_dialogs* is switched off for shared links: the dialog would open on the machine
    running the server, not on the viewer's, and block their request until someone here
    dismissed it.
    """
    import gradio as gr

    configure_model_cache()
    session = session or Session()
    can_browse = native_dialogs and dialogs_available()
    start = defaults()
    widgets: dict[str, object] = {}
    control_keys = [k.key for k in KNOBS]

    # ------------------------------------------------------------------ handlers

    def browse_rgb(current):
        picked = pick_open("Select the RGB clip, or any frame of a sequence", current)
        return collapse_to_sequence(picked) or current

    def browse_depth(current):
        picked = pick_open("Select the depth map, or any frame of a sequence", current)
        return collapse_to_sequence(picked) or current

    def browse_out(current, depth_current):
        hint = current or suggest_output(depth_current)
        # Frames in, frames out: a sequence needs a destination folder, not a filename.
        if depth_current and Path(depth_current).is_dir():
            return pick_directory("Choose a folder for the corrected frames", hint) or current
        return pick_save("Save the corrected depth map as", hint) or current

    def on_load(rgb_file, depth_file, out_current):
        if not rgb_file or not depth_file:
            return ("Select both an RGB clip and its depth map.", gr.update(), out_current)
        info = session.load(rgb_file, depth_file)
        # The output box always follows the pair that was just loaded, so Render is one
        # click. Keeping an existing entry instead leaves the path from the previous clip
        # sitting there after you load the next one - pointing the render at the wrong
        # file, under a name that says it belongs to the reel you just finished.
        return info, gr.update(maximum=session.max_frame(), value=0), suggest_output(depth_file)

    def on_profile(profile):
        """Switching profile resets the controls to that profile's defaults.

        Without this the profile would be inert: every widget always holds a value, so the
        controls would override the preset the instant it was chosen.

        A profile moves detection settings, so the cached masks no longer describe what the
        controls now say - they are dropped here and the redraw chained after this recomputes.
        """
        session.invalidate_masks()
        wanted = profile_defaults(profile)
        return [gr.update(value=wanted[k.key]) for k in CONFIG_KNOBS]

    def _painted_code(cfg, depth_info) -> float:
        from .composite import brightness_to_code, resolve_range

        return brightness_to_code(cfg.composite.brightness, depth_info.bit_depth,
                                  resolve_range(cfg.composite, depth_info.color_range))

    def render_frame(index, recompute, *values, progress=gr.Progress()):
        if session.rgb_path is None:
            return None, "Load a clip pair first."
        from .pipeline import context_frames

        cfg = build_config(dict(zip(control_keys, values)))
        index = int(index)
        if recompute:
            session.invalidate_masks()

        # Asking for one frame is never one frame's work - the persistence and temporal
        # gates need the frames either side - so the bar counts those, and says which stage
        # it is in. The first run also has to load the models, which is the longest silence
        # of all if nothing says so.
        tick = None
        if not session.has_mask(index):
            if session.detectors is None:
                progress(0.02, desc="loading detection models (first run fetches weights)")
                session.get_detectors(cfg)
            total = context_frames(cfg, index)

            def tick(done: int) -> None:  # noqa: F811 - deliberately shadowing the None
                progress(min(0.90, 0.05 + 0.85 * done / total),
                         desc=f"detecting text — frame {done} of {total}")

        mask_u8, detections = session.mask_for(index, cfg, progress=tick)
        progress(0.95, desc="compositing depth")
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
                 f"peak strength {float(alpha_rgb.max()):.2f} | painted code "
                 f"{_painted_code(cfg, session.depth_info):.0f} of "
                 f"{(1 << session.depth_info.bit_depth) - 1}")
        return sheet[..., ::-1], stats  # BGR -> RGB for gradio

    def render_clip(out_path, max_frames, *values, progress=gr.Progress()):
        """Render the whole clip and report where it went.

        The result is deliberately *not* offered as a download. Gradio serves a file by
        copying it into its own cache, which for a feature-length depth map means duplicating
        gigabytes to hand back something already sitting at a path the user chose - and it
        refuses outright for anything outside the working directory or the temp folder, which
        is most places anyone would actually write to.
        """
        if session.rgb_path is None:
            return "Load a clip pair first."
        from .pipeline import run_fix

        cfg = build_config(dict(zip(control_keys, values)))
        out = out_path.strip() if out_path else ""
        if not out:
            source = Path(session.depth_path)
            out = str(Path(tempfile.gettempdir()) / f"{source.stem}_fixed")
            if not source.is_dir():
                out += source.suffix or ".mp4"
        limit = int(max_frames) if max_frames and int(max_frames) > 0 else None
        total = limit or (session.max_frame() + 1) or None

        def tick(n: int) -> None:
            if total:
                progress(min(1.0, n / total), desc=f"rendering — frame {n} of {total}")

        progress(0.0, desc="loading detection models" if session.detectors is None
                 else "starting")
        try:
            result = run_fix(session.rgb_path, session.depth_path, out, cfg,
                             max_frames=limit, on_render=tick)
        except Exception as exc:  # noqa: BLE001 - the app must not die on a bad render
            # Full traceback to the console for diagnosis; one readable line in the app.
            traceback.print_exc()
            return f"**Render failed.** {type(exc).__name__}: {exc}"

        target = Path(out)
        what = "frames in" if target.is_dir() else "frames to"
        note = "  \n".join(f"note: {n}" for n in result["notes"])
        return f"Wrote {result['frames']} {what} `{out}`" + (f"  \n{note}" if note else "")

    # ------------------------------------------------------------------- layout

    browse_tip = "" if can_browse else (
        "  \n*Browse buttons need tkinter, which this Python does not have - paste paths "
        "instead.*" if native_dialogs else
        "  \n*Browse buttons are disabled on a shared link: the dialog would open on the "
        "machine running the server. Paste paths instead.*"
    )

    def lay_out(group: str) -> None:
        title, collapsed = GROUP_TITLES[group]
        knobs = group_knobs(group)
        if not knobs:
            return
        if collapsed:
            with gr.Accordion(title, open=False):
                for knob in knobs:
                    widgets[knob.key] = make_component(gr, knob, start[knob.key])
        else:
            gr.Markdown(f"### {title}")
            for knob in knobs:
                widgets[knob.key] = make_component(gr, knob, start[knob.key])

    with gr.Blocks(title="depth-subtitle-fixer") as demo:
        gr.Markdown(
            "## depth-subtitle-fixer\n"
            "Load an RGB clip and its DepthCrafter depth map - either a video file or a "
            "folder of frames - tune the mask and the painted grey level on one frame, then "
            "render the whole clip." + browse_tip
        )
        with gr.Row():
            rgb_file = gr.Textbox(label="RGB clip (file or frame folder)",
                                  placeholder=r"F:\clips\movie.mp4", scale=8)
            rgb_browse = gr.Button("Browse…", scale=0, min_width=110,
                                   elem_classes="dsf-browse", interactive=can_browse)
            depth_file = gr.Textbox(label="Depth map (file or frame folder)",
                                    placeholder=r"F:\clips\movie_depth.mp4", scale=8)
            depth_browse = gr.Button("Browse…", scale=0, min_width=110,
                                     elem_classes="dsf-browse", interactive=can_browse)
            load_btn = gr.Button("Load", variant="primary", scale=0, min_width=100,
                                 elem_classes="dsf-browse")
        info_box = gr.Textbox(label="Streams", lines=4, interactive=False)

        with gr.Row():
            with gr.Column(scale=1):
                lay_out("detect")
                recompute = gr.Checkbox(
                    value=False,
                    label="Recompute mask (needed after changing detection settings)")
                lay_out("detect_adv")
                lay_out("strokes_adv")
                lay_out("repair")

            with gr.Column(scale=2):
                frame_idx = gr.Slider(0, 1, value=0, step=1, label="Frame")
                sheet = gr.Image(label="source | mask | depth before | depth after",
                                 type="numpy", height=620)
                stats = gr.Markdown()
                with gr.Row():
                    out_path = gr.Textbox(
                        label="Output path (file, or folder for frames)",
                        placeholder="leave blank for a temp file", scale=8)
                    out_browse = gr.Button("Browse…", scale=0, min_width=110,
                                           elem_classes="dsf-browse", interactive=can_browse)
                    # Kept to one line: a wrapping label makes this block taller than the
                    # textbox beside it, and buttons align to the tallest thing in the row.
                    max_frames = gr.Number(label="Max frames", value=None, scale=2,
                                           placeholder="all")
                    render_btn = gr.Button("Render full clip", variant="primary", scale=0,
                                           min_width=150, elem_classes="dsf-browse")
                # Directly under the button, and never empty. Gradio draws a progress bar
                # over the output component, so this is where the bar appears - put it below
                # the Encoding panel and it lands off the bottom of a tall page, which is
                # indistinguishable from having no progress at all.
                render_status = gr.Markdown("Ready to render.")
                lay_out("encode")

        control_widgets = [widgets[key] for key in control_keys]
        live_widgets = [widgets[k.key] for k in KNOBS if k.live]

        rgb_browse.click(browse_rgb, rgb_file, rgb_file)
        depth_browse.click(browse_depth, depth_file, depth_file)
        out_browse.click(browse_out, [out_path, depth_file], out_path)
        load_btn.click(on_load, [rgb_file, depth_file, out_path],
                       [info_box, frame_idx, out_path])
        preview_inputs = [frame_idx, recompute, *control_widgets]

        # Listen on `input`, not `change`. `change` also fires when a value is set
        # programmatically, so resetting the controls for a new profile would kick off a
        # redraw per control it touched - dozens of them, each recompositing the frame,
        # before anything appeared. `input` fires only when the user moves something.
        #
        # `always_last` coalesces a drag: the run in flight finishes, then one more runs with
        # wherever the slider ended up, instead of queueing a render per step.
        for control in [frame_idx, recompute, *live_widgets]:
            control.input(render_frame, preview_inputs, [sheet, stats],
                          trigger_mode="always_last")

        # One redraw after the profile has finished rewriting the other controls.
        widgets["profile"].change(
            on_profile, widgets["profile"], [widgets[k.key] for k in CONFIG_KNOBS],
        ).then(render_frame, preview_inputs, [sheet, stats])

        render_btn.click(render_clip, [out_path, max_frames, *control_widgets],
                         render_status)

    return demo


def launch(share: bool = False, port: int = 7860) -> None:
    # Gradio 6 takes styling at launch time rather than on the Blocks constructor.
    build_app(native_dialogs=not share).launch(
        share=share, server_port=port, inbrowser=True, css=_CSS,
    )
