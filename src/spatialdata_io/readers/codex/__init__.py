"""Read CODEX regional images and cell tables.

The public entry point :func:`codex` accepts flat regional exports, Processor
marker images, Uploader montages and CODEX Instrument Manager QPI scans.
Cells can annotate an explicitly registered instance mask or centroid circles.
"""

from __future__ import annotations

from spatialdata_io.readers.codex._reader import codex

__all__ = ["codex"]
