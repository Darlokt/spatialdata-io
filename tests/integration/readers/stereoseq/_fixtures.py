"""Independent synthetic SAW contracts; no fixture claims an observed producer run.

Path spellings follow the pinned compact workflow:
https://github.com/STOmics/SAW/blob/41e70aa12bb16da8c9ddae344863cef2ca12f735/script/stereoPipeline.sh
and the versioned SAW 7/8 manuals. Layout family is not a release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import tifffile
from scipy.sparse import csr_array

if TYPE_CHECKING:
    from pathlib import Path

type Family = Literal["compact_numbered", "numbered", "outs", "flat"]
type Profile = Literal["name_only", "id_and_name"]
type Product = Literal[
    "square",
    "cells",
    "adjusted_cells",
    "analysis",
    "adjusted_analysis",
    "registered",
    "mask",
    "tissue_cut",
    "expression",
    "expression_tissue_cut",
]

CELL_IDS = (200, 100)
CELL_COORDINATES = ((10, 20), (30, 40))
FEATURE_NAMES = ("z", "a")
FEATURE_IDS = ("g_z", "g_a")
CELL_COUNTS = ((11, 0), (0, 22))
CELL_EXONS = ((5, 0), (0, 6))
SQUARE_COORDINATES = ((5, 6), (1, 2))


@dataclass(frozen=True, slots=True)
class DatasetFixture:
    """Carry explicit artifacts without rediscovering vendor layout semantics."""

    root: Path
    sample_id: str
    square: Path
    cells: Path
    adjusted_cells: Path
    analysis: Path
    adjusted_analysis: Path
    image_directory: Path


def write_square_gef(
    path: Path,
    *,
    profile: Profile = "name_only",
    bins: tuple[int, ...] = (1, 50),
    overflow: bool = False,
    empty_feature: bool = False,
) -> Path:
    """Write source-ordered z,a features, first-occurrence coordinates and exons."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = np.uint32(2 if profile == "name_only" else 4)
        handle.attrs["offsetX"] = np.int32(0)
        handle.attrs["offsetY"] = np.int32(0)
        for size in bins:
            group = handle.create_group(f"geneExp/bin{size}")
            genes: list[tuple[bytes, int, int]] | list[tuple[bytes, bytes, int, int]]
            if profile == "name_only":
                fields = [("gene", "S16"), ("offset", "u4"), ("count", "u4")]
                genes = [(b"z", 0, 2), (b"a", 2, 1)]
                if empty_feature:
                    genes.append((b"empty", 3, 0))
            else:
                fields = [("geneID", "S16"), ("geneName", "S16"), ("offset", "u4"), ("count", "u4")]
                genes = [(b"g_z", b"z", 0, 2), (b"g_a", b"a", 2, 1)]
                if empty_feature:
                    genes.append((b"g_empty", b"empty", 3, 0))
            group["gene"] = np.array(genes, dtype=fields)
            first, second = (200, 100) if overflow else (2, 4)
            dataset = group.create_dataset(
                "expression",
                data=np.array(
                    [(5, 6, first), (5, 6, second), (1, 2, 3)], dtype=[("x", "i4"), ("y", "i4"), ("count", "u1")]
                ),
                chunks=(2,),
            )
            dataset.attrs["resolution"] = np.uint32(500)
            dataset.attrs["minX"] = np.int32(0)
            dataset.attrs["minY"] = np.int32(0)
            dataset.attrs["maxX"] = np.int32(99)
            dataset.attrs["maxY"] = np.int32(99)
            group["exon"] = np.array([1, 2, 1], dtype="u1")
    return path


def write_cell_gef(path: Path, *, profile: Profile = "name_only", adjusted: bool = False) -> Path:
    """Write native cell spans, noncontiguous source IDs and paired borders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    shift = 1 if adjusted else 0
    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = np.uint32(2 if profile == "name_only" else 4)
        handle.attrs["offsetX"] = np.int32(0)
        handle.attrs["offsetY"] = np.int32(0)
        handle.attrs["resolution"] = np.uint32(500)
        group = handle.create_group("cellBin")
        cell_fields = ("id", "x", "y", "offset", "geneCount", "expCount", "dnbCount", "area", "cellTypeID", "clusterID")
        group["cell"] = np.array(
            [(200, 10 + shift, 20, 0, 1, 11, 1, 9, 0, 1), (100, 30 + shift, 40, 1, 1, 22, 1, 16, 1, 0)],
            dtype=[(field, "u4") for field in cell_fields],
        )
        genes: list[tuple[bytes, int, int, int, int]] | list[tuple[bytes, bytes, int, int, int, int]]
        if profile == "name_only":
            fields = [
                ("geneName", "S16"),
                ("offset", "u4"),
                ("cellCount", "u4"),
                ("expCount", "u4"),
                ("maxMIDcount", "u4"),
            ]
            genes = [(b"z", 0, 1, 11, 11), (b"a", 1, 1, 22, 22)]
        else:
            fields = [
                ("geneID", "S16"),
                ("geneName", "S16"),
                ("offset", "u4"),
                ("cellCount", "u4"),
                ("expCount", "u4"),
                ("maxMIDcount", "u4"),
            ]
            genes = [(b"g_z", b"z", 0, 1, 11, 11), (b"g_a", b"a", 1, 1, 22, 22)]
        group["gene"] = np.array(genes, dtype=fields)
        group["cellExp"] = np.array(
            [(0, 11), (1, 22)], dtype=[("geneID", "u2" if profile == "name_only" else "u4"), ("count", "u2")]
        )
        group["geneExp"] = np.array([(0, 11), (1, 22)], dtype=[("cellID", "u4"), ("count", "u2")])
        group["cellExon"] = np.array([5, 6], dtype="u4")
        group["geneExon"] = np.array([5, 6], dtype="u4")
        group["cellExpExon"] = np.array([5, 6], dtype="u2")
        group["geneExpExon"] = np.array([5, 6], dtype="u2")
        group["blockIndex"] = np.array([0, 2], dtype="u4")
        group["blockSize"] = np.array([100, 100, 1, 1], dtype="u4")
        group["cellTypeList"] = np.array([b"unknown", b"cell"], dtype="S16")
        borders = np.full((2, 32, 2), 32767, dtype="i2")
        borders[:, :4, :] = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype="i2")
        group["cellBorder"] = borders
    return path


def write_analysis(  # noqa: PLR0913, RUF100 - explicit fixture variants.
    path: Path,
    *,
    profile: Profile = "name_only",
    reordered: bool = False,
    mismatched: bool = False,
    adjusted: bool = False,
    sample_id: str = "sample",
) -> Path:
    """Write independently indexed analysis with intentionally nonraw values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    source_ids = np.array(CELL_IDS, dtype=np.uint32)
    features = list(FEATURE_NAMES if profile == "name_only" else FEATURE_IDS)
    matrix = np.array([[1.25, 2.5], [3.75, 5.0]], dtype=np.float32)
    coordinates = np.array(CELL_COORDINATES, dtype=np.int64)
    coordinates[:, 0] += int(adjusted)
    cells = [str((int(x) << 32) | int(y)) for x, y in coordinates]
    if reordered:
        cells.reverse()
        features.reverse()
        matrix = matrix[::-1, ::-1]
        coordinates = coordinates[::-1]
        source_ids = source_ids[::-1]
    if mismatched:
        cells = ["unrelated_A", "unrelated_B"]
        features = ["unrelated_gene_A", "unrelated_gene_B"]
        source_ids = np.array([900, 901], dtype=np.uint32)
    obs = pd.DataFrame(
        {
            "id": source_ids,
            "x": coordinates[:, 0].astype(np.uint32),
            "y": coordinates[:, 1].astype(np.uint32),
            "analysis_annotation": pd.array(["first", "second"], dtype="string"),
        },
        index=cells,
    )
    annotation = ad.AnnData(
        csr_array(matrix),
        obs=obs,
        var=pd.DataFrame({"analysis_feature": [9, 8]}, index=features),
    )
    annotation.obsm["spatial"] = coordinates.astype(np.uint32)
    annotation.obsm["X_pca"] = np.array([[1, 2], [3, 4]], dtype=np.float32)
    annotation.uns["bin_type"] = "cell_bins"
    annotation.uns["bin_size"] = 1
    annotation.uns["sn"] = pd.DataFrame({"batch": [0], "sn": [sample_id]})
    annotation.uns["resolution"] = 500
    annotation.write_h5ad(path)
    return path


def write_dataset(  # noqa: PLR0913, RUF100 - explicit fixture variants.
    root: Path,
    *,
    family: Family = "compact_numbered",
    profile: Profile = "name_only",
    sample_id: str = "sample",
    products: tuple[Product, ...] = ("square", "cells", "analysis"),
    bins: tuple[int, ...] = (1, 50),
    overflow: bool = False,
) -> DatasetFixture:
    """Create only selected products in one documented structural family."""
    if family == "compact_numbered":
        square_dir, cell_dir, analysis_dir, image_dir = "04.tissuecut", "041.cellcut", "051.cellcluster", "03.register"
        count_dir = "02.count"
    elif family == "numbered":
        square_dir, cell_dir, analysis_dir, image_dir = "05.tissuecut", "051.cellcut", "061.cellcluster", "04.register"
        count_dir = "03.count"
    elif family == "outs":
        square_dir = cell_dir = "outs/feature_expression"
        analysis_dir, image_dir, count_dir = "outs/analysis", "outs/image", "outs/feature_expression"
    else:
        square_dir = cell_dir = analysis_dir = image_dir = count_dir = "."

    modern = family == "outs"
    square_name = f"{sample_id}.gef" if family == "compact_numbered" else f"{sample_id}.tissue.gef"
    analysis_name = f"{sample_id}.cellbin_0.5.h5ad" if modern else f"{sample_id}.cell.cluster.h5ad"
    adjusted_name = f"{sample_id}.cellbin_0.5.adjusted.h5ad" if modern else f"{sample_id}.adjusted.cell.cluster.h5ad"
    result = DatasetFixture(
        root=root,
        sample_id=sample_id,
        square=root / square_dir / square_name,
        cells=root / cell_dir / f"{sample_id}.cellbin.gef",
        adjusted_cells=root / cell_dir / f"{sample_id}.adjusted.cellbin.gef",
        analysis=root / analysis_dir / analysis_name,
        adjusted_analysis=root / analysis_dir / adjusted_name,
        image_directory=root / image_dir,
    )

    # Empty legacy directories are needed by its unconditional scans, not by the new product contract.
    for directory in (square_dir, cell_dir, analysis_dir, image_dir, count_dir):
        (root / directory).mkdir(parents=True, exist_ok=True)

    if "square" in products:
        write_square_gef(result.square, profile=profile, bins=bins, overflow=overflow)
    if "cells" in products:
        write_cell_gef(result.cells, profile=profile)
    if "adjusted_cells" in products:
        write_cell_gef(result.adjusted_cells, profile=profile, adjusted=True)
    if "analysis" in products:
        write_analysis(result.analysis, profile=profile, sample_id=sample_id)
    if "adjusted_analysis" in products:
        write_analysis(result.adjusted_analysis, profile=profile, adjusted=True, sample_id=sample_id)
    _write_images(result, products, family=family, profile=profile)
    return result


def _write_images(fixture: DatasetFixture, products: tuple[Product, ...], *, family: Family, profile: Profile) -> None:
    """Write distinct image roles with deterministic pixels and codec variants."""
    sample_id = fixture.sample_id
    if family == "compact_numbered":
        image_prefix = sample_id
    elif family in ("outs", "flat"):
        image_prefix = f"{sample_id}_HE"
    else:
        image_prefix = f"HE_{sample_id}"

    suffixes: dict[Product, str] = {
        "registered": f"{image_prefix}_regist.tif",
        "mask": f"{image_prefix}_mask.tif",
        "tissue_cut": f"{image_prefix}_tissue_cut.tif",
        "expression": "bin1_img.tif",
        "expression_tissue_cut": "bin1_img_tissue_cut.tif",
    }
    for product in products:
        if product in suffixes:
            pixels = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
            if product in ("mask", "expression_tissue_cut"):
                pixels = (pixels % 3 == 0).astype(np.uint8)
            tifffile.imwrite(
                fixture.image_directory / suffixes[product],
                pixels,
                metadata={"axes": "YX"},
                compression="deflate" if profile == "id_and_name" else None,
            )
