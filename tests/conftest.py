"""Shared fixtures. Every test builds its own synthetic media - no sample files needed."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs model weights or a full encode pass")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run tests marked slow")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="needs --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for name in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def gradient_background(width: int, height: int) -> np.ndarray:
    """A smooth non-uniform background - nothing here should look like a glyph."""
    xs = np.linspace(30, 170, width, dtype=np.float32)[None, :]
    ys = np.linspace(20, 90, height, dtype=np.float32)[:, None]
    base = np.clip(xs + ys, 0, 255)
    rgb = np.stack([base, base * 0.8 + 20, base * 0.6 + 40], axis=-1)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def draw_subtitle(background: np.ndarray, text: str = "HELLO WORLD",
                  font_size: int = 40, y_frac: float = 0.82,
                  fill=(255, 255, 255), outline=(0, 0, 0), stroke: int = 3
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Burn subtitle-style text into an RGB frame.

    Returns ``(frame, fill_mask)``. ``fill_mask`` marks the glyph fill only. The full
    rendered text - fill *plus* its outline, which is what the tool aims to mask, since
    DepthCrafter corrupts the depth under both - is available via :func:`draw_subtitle_full`.
    """
    frame, fill, _ = draw_subtitle_full(background, text, font_size, y_frac, fill, outline,
                                        stroke)
    return frame, fill


def draw_subtitle_full(background: np.ndarray, text: str = "HELLO WORLD",
                       font_size: int = 40, y_frac: float = 0.82,
                       fill=(255, 255, 255), outline=(0, 0, 0), stroke: int = 3
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """As :func:`draw_subtitle`, also returning the fill+outline mask."""
    h, w = background.shape[:2]
    img = Image.fromarray(background.copy())
    draw = ImageDraw.Draw(img)
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2
    y = int(h * y_frac) - th // 2
    draw.text((x, y), text, font=font, fill=fill, stroke_width=stroke, stroke_fill=outline)

    fill_img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(fill_img).text((x, y), text, font=font, fill=255)
    full_img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(full_img).text((x, y), text, font=font, fill=255,
                                  stroke_width=stroke, stroke_fill=255)
    return np.array(img), np.array(fill_img), np.array(full_img)


def text_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


@pytest.fixture
def subtitle_frame():
    bg = gradient_background(640, 360)
    return draw_subtitle(bg, "HELLO WORLD", font_size=36)


@pytest.fixture
def tmp_media(tmp_path):
    return tmp_path
