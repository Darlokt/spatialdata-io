"""Top-level command group."""

from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Convert supported spatial-omics formats to SpatialData."""
