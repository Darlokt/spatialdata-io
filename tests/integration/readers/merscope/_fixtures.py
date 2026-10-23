"""Deterministic standard and VPT MERSCOPE cohort input profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile
from shapely.geometry import MultiPolygon, Polygon

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class MerscopeFixture:
    """Identify one independently written instrument/VPT profile."""

    root: Path
    analysis: Path
    profile: Literal["standard", "cellpose", "watershed"]
    cell_ids: tuple[str, ...]


def _write_tables(path: Path, ids: tuple[str, ...], *, vpt: bool, extra_gene: bool) -> None:
    """Keep count order and named annotation values deterministic."""
    increment = 100 if vpt else 0
    counts = pd.DataFrame(
        {
            "GeneB": [1 + increment, 0, 3 + increment],
            "Blank-1": [2 + increment, 5 + increment, 0],
            "GeneA": [0, 4 + increment, 1 + increment],
            "ZeroGene": [0, 0, 0],
        },
        index=pd.Index(ids, name="cell"),
    )
    if extra_gene:
        counts["BlanketGene"] = [7, 8, 9]
        counts["blank-2"] = [10, 11, 12]
    counts.to_csv(path / "cell_by_gene.csv")
    pd.DataFrame(
        {
            "center_x": [2.0, 10.0, 21.0],
            "center_y": [2.0, 1.0, 1.0],
            "volume": [12.0, 4.0, 8.0],
            "annotation": ["A", "B", "C"],
        },
        index=pd.Index(ids, name="EntityID"),
    ).to_csv(path / "cell_metadata.csv")


def _write_boundaries(path: Path, ids: tuple[str, ...], *, incomplete_z0: bool) -> None:
    """Exercise holes, discontinuous shapes, and cells absent from the first z plane."""
    outer = [(0, 0), (4, 0), (4, 4), (0, 4)]
    hole = [(1, 1), (1, 3), (3, 3), (3, 1)]
    island = Polygon([(6, 0), (7, 0), (7, 1), (6, 1)])
    geometries = [
        MultiPolygon([Polygon(outer, [hole]), island]),
        MultiPolygon([Polygon([(0, 0), (2, 0), (2, 6), (0, 6)])]),
        MultiPolygon([Polygon([(9, 0), (11, 0), (11, 2), (9, 2)])]),
        MultiPolygon([Polygon([(20, 0), (22, 0), (22, 2), (20, 2)])]),
    ]
    frame = gpd.GeoDataFrame(
        {
            "ID": [0, 1, 2, 3],
            "EntityID": [int(ids[0]), int(ids[0]), int(ids[1]), int(ids[2])],
            "Type": ["Cell"] * 4,
            "ZIndex": [0, 2, 2 if incomplete_z0 else 0, 0],
            "ZLevel": [1.5, 4.5, 4.5 if incomplete_z0 else 1.5, 1.5],
            "Geometry": geometries,
            "unused": ["ignored"] * 4,
        },
        geometry="Geometry",
    )
    frame.to_parquet(path)


def write_merscope_dataset(
    path: Path,
    *,
    profile: Literal["standard", "cellpose", "watershed"] = "standard",
    incomplete_z0: bool = False,
    extra_gene: bool = False,
    reassigned_transcripts: bool = False,
) -> MerscopeFixture:
    """Write a complete standard root and optional distinct VPT analysis."""
    root = path / "slide" / "region"
    images = root / "images"
    images.mkdir(parents=True)
    np.savetxt(images / "micron_to_mosaic_pixel_transform.csv", [[2.0, 0.25, 10.0], [0.0, 3.0, -7.0], [0.0, 0.0, 1.0]])
    # Deliberately create files in an order differing from lexical stain order.
    for stain, offset in (("DAPI", 0), ("Cyto", 10_000)):
        for z in (2, 0):
            image = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80) + offset + z * 100
            tifffile.imwrite(
                images / f"mosaic_{stain}_z{z}.tif",
                image,
                photometric="minisblack",
                metadata={"axes": "YX"},
                rowsperstrip=8,
                compression="deflate" if stain == "Cyto" else None,
            )
    original_ids = ("1001", "2002", "3003")
    _write_tables(root, original_ids, vpt=False, extra_gene=extra_gene)
    _write_boundaries(root / "cell_boundaries.parquet", original_ids, incomplete_z0=incomplete_z0)
    pd.DataFrame(
        {
            "barcode_id": [0, 1, 2, 3],
            "global_x": [2.0, 10.0, 21.0, 8.0],
            "global_y": [2.0, 1.0, 1.0, 6.0],
            "global_z": [0.0, 2.0, 2.0, 0.0],
            "x": [20.0, 30.0, 40.0, 50.0],
            "y": [21.0, 31.0, 41.0, 51.0],
            "fov": [0, 0, 1, 1],
            "gene": ["GeneB", "GeneA", "Blank-1", "ZeroGene"],
            "transcript_id": [10, 20, 30, 40],
            "cell_id": [1001, 2002, 3003, -1],
        },
        index=pd.Index([7, 9, 7, 10], name=""),
    ).to_csv(root / "detected_transcripts.csv")
    if profile == "standard":
        return MerscopeFixture(root, root, profile, original_ids)
    analysis = root / "vpt"
    analysis.mkdir()
    ids = ("9001", "9002", "9003")
    _write_tables(analysis, ids, vpt=True, extra_gene=extra_gene)
    _write_boundaries(analysis / f"{profile}_micron_space.parquet", ids, incomplete_z0=incomplete_z0)
    if reassigned_transcripts:
        transcripts = pd.read_csv(root / "detected_transcripts.csv", index_col=0)
        transcripts["cell_id"] = [9001, 9002, 9003, -1]
        transcripts.to_csv(analysis / "detected_transcripts.csv")
    return MerscopeFixture(root, analysis, profile, ids)
