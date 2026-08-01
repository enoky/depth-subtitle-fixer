"""Command line interface.

    dsf probe   <video>                     inspect a stream
    dsf detect  --rgb ...  --out-mask ...   detection only, cache the masks
    dsf render  --depth ... --mask ...      composite cached masks at a chosen brightness
    dsf fix     --rgb ... --depth ...       end to end
    dsf preview --rgb ... --depth ...       contact sheets for chosen frames
    dsf ui                                  local Gradio app
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import warnings
from pathlib import Path

from . import __version__
from .config import PipelineConfig, apply_profile, configure_model_cache
from .detect.doctr_det import ARCHS as DOCTR_ARCHS

#: (argparse dest, config section, field name). Any flag left as None keeps the profile value.
ARG_MAP: list[tuple[str, str, str]] = [
    ("detectors", "detect", "detectors"),
    ("det_arch", "detect", "det_arch"),
    ("min_score", "detect", "min_score"),
    ("batch_size", "detect", "batch_size"),
    ("device", "detect", "device"),
    ("detect_every", "detect", "detect_every"),
    ("roi", "filters", "roi"),
    ("min_text_height", "filters", "min_text_height"),
    ("max_text_height", "filters", "max_text_height"),
    ("scene_text", "filters", "scene_text"),
    ("min_contrast", "filters", "min_contrast"),
    ("min_persist_frames", "filters", "min_persist_frames"),
    ("persist_window", "filters", "persist_window"),
    ("pad", "strokes", "pad"),
    ("polarity", "strokes", "polarity"),
    ("min_cc_area", "strokes", "min_cc_area"),
    ("max_stroke", "strokes", "max_stroke"),
    ("min_response", "strokes", "min_response"),
    ("background_scale", "strokes", "background_scale"),
    ("luma_tol", "strokes", "luma_tol"),
    ("rim_expand", "strokes", "rim_expand"),
    ("temporal", "temporal", "mode"),
    ("temporal_window", "temporal", "window"),
    ("brightness", "composite", "brightness"),
    ("brightness_mode", "composite", "brightness_mode"),
    ("dilate", "composite", "dilate"),
    ("feather", "composite", "feather"),
    ("heal", "composite", "heal"),
    ("heal_scope", "composite", "heal_scope"),
    ("heal_dilate", "composite", "heal_dilate"),
    ("range", "composite", "value_range"),
    ("encoder", "encode", "encoder"),
    ("crf", "encode", "crf"),
    ("preset", "encode", "preset"),
    ("lossless", "encode", "lossless"),
]


def add_detect_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("detection")
    g.add_argument("--profile", default="subtitles", choices=("subtitles", "credits", "both"),
                   help="preset tuned for static subtitles, scrolling credits, or both")
    g.add_argument("--detectors", type=lambda s: tuple(x.strip() for x in s.split(",")),
                   help="comma separated: doctr,easyocr (default: both)")
    g.add_argument("--det-arch", choices=DOCTR_ARCHS,
                   help="docTR detection architecture (default: db_resnet50)")
    g.add_argument("--min-score", type=float, help="minimum detector confidence")
    g.add_argument("--batch-size", type=int, help="frames per detector forward pass")
    g.add_argument("--device", help="cuda, cpu, or auto")
    g.add_argument("--detect-every", type=int,
                   help="run detection every Nth frame and propagate in between")

    g = p.add_argument_group("text filtering")
    g.add_argument("--roi", help="full | bottom:0.30 | top:0.20 | x0,y0,x1,y1 (normalised)")
    g.add_argument("--min-text-height", type=float, help="as a fraction of frame height")
    g.add_argument("--max-text-height", type=float, help="as a fraction of frame height")
    g.add_argument("--scene-text", choices=("keep", "mask"),
                   help="keep: leave filmed text alone (default). mask: mask all text found")
    g.add_argument("--min-contrast", type=float, help="minimum P95-P50 luma spread in a box")
    g.add_argument("--min-persist-frames", type=int,
                   help="frames a detection must survive to count as an overlay")
    g.add_argument("--persist-window", type=int, help="frames examined for persistence")

    g = p.add_argument_group("glyph extraction")
    g.add_argument("--pad", type=int, help="pixels of padding around each detection box")
    g.add_argument("--polarity", choices=("auto", "light", "dark"),
                   help="are the glyphs brighter or darker than their background")
    g.add_argument("--min-cc-area", type=int, help="drop connected components below this area")
    g.add_argument("--max-stroke", type=float,
                   help="floor for the stroke-thickness cap in px (it also scales with text size)")
    g.add_argument("--min-response", type=float,
                   help="minimum stroke contrast for a box to count as text (default 0.05)")
    g.add_argument("--background-scale", type=float,
                   help="background window as a fraction of text height; raise it if heavy "
                        "or bold text comes back hollow (default 0.9)")
    g.add_argument("--luma-tol", type=float,
                   help="colour tolerance when following a glyph's outline (fill uses opacity)")
    g.add_argument("--rim-expand", type=int,
                   help="px the mask grows from the glyph core into its outline (default 3)")
    g.add_argument("--temporal", choices=("median", "max", "none"),
                   help="temporal mask filter")
    g.add_argument("--temporal-window", type=int, help="frames in the temporal filter")


def add_composite_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("depth repair")
    g.add_argument("--brightness", type=float,
                   help="0..1 grey level for the painted text, within the legal code range")
    g.add_argument("--brightness-mode", choices=("absolute", "relative"),
                   help="absolute: a constant plane. relative: just in front of local depth")
    g.add_argument("--dilate", type=int, help="grow the glyph mask by N px")
    g.add_argument("--feather", type=float, help="gaussian sigma applied to the mask edge")
    g.add_argument("--heal", choices=("edt", "none"),
                   help="repair the corrupted depth around the glyphs before painting")
    g.add_argument("--heal-scope", choices=("glyph", "region"),
                   help="glyph: a halo around the strokes. region: the whole detection box")
    g.add_argument("--heal-dilate", type=int, help="halo radius in px for --heal-scope glyph")
    g.add_argument("--range", choices=("auto", "tv", "pc"),
                   help="luma code range of the depth map (auto reads the stream tags)")

    g = p.add_argument_group("encoding")
    g.add_argument("--encoder", choices=("libx265", "libx264", "ffv1"))
    g.add_argument("--crf", type=int, help="quality, lower is better (default 12)")
    g.add_argument("--preset", help="encoder preset (default slow)")
    g.add_argument("--lossless", action="store_true", default=None)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Profile first, then any explicitly supplied flag wins."""
    cfg = PipelineConfig()
    profile = getattr(args, "profile", None)
    if profile:
        cfg = apply_profile(cfg, profile)

    updates: dict[str, dict] = {}
    for dest, section, field in ARG_MAP:
        value = getattr(args, dest, None)
        if value is not None:
            updates.setdefault(section, {})[field] = value
    for section, values in updates.items():
        cfg = dataclasses.replace(cfg, **{section: dataclasses.replace(
            getattr(cfg, section), **values)})
    return cfg


def _progress(description: str, total: int | None):
    from rich.progress import (BarColumn, Progress, SpinnerColumn, TaskProgressColumn,
                               TextColumn, TimeRemainingColumn)

    columns = [SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
               TaskProgressColumn(), TextColumn("{task.completed} frames"),
               TimeRemainingColumn()]
    progress = Progress(*columns)
    task = progress.add_task(description, total=total if total and total > 0 else None)
    return progress, task


def parse_frame_list(spec: str) -> list[int]:
    """Parse '12,40,100' or '0-120:20' (start-end:step) into frame indices."""
    out: list[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            rng, _, step = part.partition(":")
            lo, _, hi = rng.partition("-")
            out.extend(range(int(lo), int(hi) + 1, int(step) if step else 1))
        else:
            out.append(int(part))
    return sorted(set(out))


# --------------------------------------------------------------------------- commands

def cmd_probe(args) -> int:
    from rich.console import Console
    from rich.table import Table

    from .media import probe

    console = Console()
    for path in args.videos:
        info = probe(path)
        table = Table(title=str(path), show_header=False)
        for key, value in (
            ("resolution", f"{info.width}x{info.height}"),
            ("codec", info.codec_name),
            ("pix_fmt", f"{info.pix_fmt}  (chroma {info.chroma}, {info.bit_depth}-bit)"),
            ("decode as", info.decode_pix_fmt),
            ("fps", f"{float(info.fps):.6f}"),
            ("frames", f"{info.nb_frames}{'' if info.frames_exact else ' (estimated)'}"),
            ("kind", "image sequence" if info.codec_name == "imageseq" else "video"),
            ("colour range", info.color_range),
            ("primaries/transfer/matrix",
             f"{info.color_primaries or '-'} / {info.color_transfer or '-'} / "
             f"{info.color_space or '-'}"),
        ):
            table.add_row(key, str(value))
        console.print(table)
    return 0


def cmd_detect(args) -> int:
    from .maskcache import MaskCacheWriter
    from .media import probe
    from .pipeline import iter_masks

    cfg = build_config(args)
    info = probe(args.rgb)
    total = args.max_frames or info.nb_frames
    progress, task = _progress("detecting text", total)
    with progress, MaskCacheWriter(args.out_mask, info.width, info.height, info.fps,
                                   args.rgb, cfg.to_dict()) as cache:
        for mask in iter_masks(args.rgb, cfg, info, max_frames=args.max_frames):
            cache.write(mask)
            progress.update(task, advance=1)
    print(f"wrote {cache.frames} masks to {args.out_mask}")
    return 0


def cmd_render(args) -> int:
    from .maskcache import read_masks
    from .media import probe
    from .pipeline import render_from_masks

    cfg = build_config(args)
    depth_info = probe(args.depth)
    total = probe(args.mask).nb_frames or depth_info.nb_frames
    progress, task = _progress("compositing depth", total)
    with progress:
        def tick(_n: int) -> None:
            progress.update(task, advance=1)

        written = render_from_masks(args.depth, read_masks(args.mask), args.out, cfg,
                                    depth_info, start=args.depth_offset, progress=tick)
    print(f"wrote {written} frames to {args.out}")
    return 0


def cmd_fix(args) -> int:
    from .media import probe
    from .pipeline import run_fix

    cfg = build_config(args)
    rgb_info, depth_info = probe(args.rgb), probe(args.depth)
    counts = [n for n in (rgb_info.nb_frames, depth_info.nb_frames) if n]
    total = args.max_frames or (min(counts) if counts else None)
    progress, task = _progress("detect + composite", total)

    with progress:
        def tick(_n: int) -> None:
            progress.update(task, advance=1)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = run_fix(args.rgb, args.depth, args.out, cfg,
                             mask_cache=args.mask_cache,
                             rgb_offset=args.rgb_offset, depth_offset=args.depth_offset,
                             max_frames=args.max_frames, on_render=tick)
    for w in caught:
        print(f"note: {w.message}", file=sys.stderr)
    print(f"wrote {result['frames']} frames to {args.out}")
    if args.mask_cache:
        print(f"mask cache: {args.mask_cache} (re-render instantly with `dsf render`)")
    return 0


def cmd_preview(args) -> int:
    from .preview import write_previews

    cfg = build_config(args)
    indices = parse_frame_list(args.frames)
    if not indices:
        print("no frames requested", file=sys.stderr)
        return 2
    paths = write_previews(args.rgb, args.depth, indices, cfg, args.out_dir,
                           panel_width=args.panel_width)
    for path in paths:
        print(path)
    if not paths:
        print("no previews written - are the requested frames within the clip?",
              file=sys.stderr)
        return 1
    return 0


def cmd_ui(args) -> int:
    from .ui import launch

    launch(share=args.share, port=args.port)
    return 0


# --------------------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dsf",
        description="Fix DepthCrafter depth maps corrupted by burned-in subtitles or credits.",
    )
    parser.add_argument("--version", action="version", version=f"dsf {__version__}")
    parser.add_argument("--models-dir", help="where to cache model weights (default ./models)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="show stream information")
    p.add_argument("videos", nargs="+")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("detect", help="detect text and cache the masks")
    p.add_argument("--rgb", required=True, help="source RGB clip")
    p.add_argument("--out-mask", required=True, help="mask cache to write (.mkv recommended)")
    p.add_argument("--max-frames", type=int)
    add_detect_args(p)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("render", help="composite cached masks onto the depth map")
    p.add_argument("--depth", required=True, help="DepthCrafter depth map")
    p.add_argument("--mask", required=True, help="mask cache from `dsf detect`")
    p.add_argument("--out", required=True)
    p.add_argument("--depth-offset", type=int, default=0)
    add_composite_args(p)
    p.set_defaults(func=cmd_render, profile=None)

    p = sub.add_parser("fix", help="detect and composite in one pass")
    p.add_argument("--rgb", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mask-cache", help="also save the masks here for instant re-rendering")
    p.add_argument("--rgb-offset", type=int, default=0)
    p.add_argument("--depth-offset", type=int, default=0)
    p.add_argument("--max-frames", type=int)
    add_detect_args(p)
    add_composite_args(p)
    p.set_defaults(func=cmd_fix)

    p = sub.add_parser("preview", help="write contact sheets for chosen frames")
    p.add_argument("--rgb", required=True)
    p.add_argument("--depth", required=True)
    p.add_argument("--frames", required=True, help="e.g. 120,450,900 or 0-600:60")
    p.add_argument("--out-dir", default="previews")
    p.add_argument("--panel-width", type=int, default=640)
    add_detect_args(p)
    add_composite_args(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("ui", help="launch the local Gradio app")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="expose a public gradio.live URL")
    p.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_model_cache(Path(args.models_dir) if args.models_dir else None)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
