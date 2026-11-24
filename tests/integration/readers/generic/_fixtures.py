"""Deterministic Generic source geometry fixtures."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from shapely.geometry import mapping

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry


def write_geojson(
    path: Path,
    geometries: Sequence[BaseGeometry | None],
    *,
    collection: bool = False,
    feature_ids: Sequence[object] | None = None,
) -> Path:
    values = [mapping(geometry) if geometry is not None else None for geometry in geometries]
    document = (
        {"type": "GeometryCollection", "geometries": values}
        if collection
        else {
            "type": "FeatureCollection",
            "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
            "features": [
                {"type": "Feature", "id": identifier, "properties": {"note": "discarded"}, "geometry": value}
                for identifier, value in zip(
                    feature_ids if feature_ids is not None else [f"vendor-{index}" for index in range(len(values))],
                    values,
                    strict=True,
                )
            ],
        }
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    return path
