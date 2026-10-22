"""Experimental conversions between legacy AnnData and SpatialData."""

from __future__ import annotations

from spatialdata_io.converters.legacy_anndata import (
    from_legacy_anndata,
    to_legacy_anndata,
)

__all__ = [
    "from_legacy_anndata",
    "to_legacy_anndata",
]
