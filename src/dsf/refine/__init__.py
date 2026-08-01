"""Refinement of detector boxes down to per-glyph alpha masks."""

from .strokes import AlphaPatch, extract_patch, compose_alpha

__all__ = ["AlphaPatch", "extract_patch", "compose_alpha"]
