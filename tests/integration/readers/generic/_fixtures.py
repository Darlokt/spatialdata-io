"""Deterministic Generic source geometry fixtures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shapely.geometry import mapping

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry


def write_geojson(path: Path, geometries: Sequence[BaseGeometry | None], *, collection: bool = False) -> Path:
    values = [mapping(geometry) if geometry is not None else None for geometry in geometries]
    document = (
        {"type": "GeometryCollection", "geometries": values}
        if collection
        else {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [
                {"type": "Feature", "id": f"vendor-{index}", "properties": {"note": "discarded"}, "geometry": value}
                for index, value in enumerate(values)
            ],
        }
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
