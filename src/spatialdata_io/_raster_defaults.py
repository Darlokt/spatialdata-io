"""Central pyramid and chunk defaults of every reader that returns images or labels.

This module has no dependencies, so readers (through
``readers._utils.image``) and the lightweight command-line interface share one
source. Changing a value here changes every reader signature and command.
"""

from __future__ import annotations

DEFAULT_TILE_SIZE = 1024
"""Edge of the square y/x tiles of images and labels."""

DEFAULT_CHUNKS: tuple[int, int, int] = (1, DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)
"""Image chunks in ``c, y, x`` order: one channel per chunk; labels use the y/x sizes."""

DEFAULT_SCALE_FACTORS: tuple[int, ...] = ()
"""Generated pyramid levels; empty for a single scale."""
