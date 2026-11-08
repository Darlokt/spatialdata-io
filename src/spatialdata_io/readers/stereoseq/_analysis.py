"""Import SAW/Stereopy clustering H5ADs and match them to the raw GEF products.

An analysis H5AD (``<sample>.bin<N>_<r>.h5ad``, ``<sample>.cellbin_<r>[.adjusted].h5ad``
or a historical ``*.cluster.h5ad``) is a derived table written by Stereopy::

    obs.index        str((x << 32) | y), or "<prefix>:<x>_<y>" after reindexing
    obs["x"], ["y"]  optional copies of the grid coordinates
    obs["id"]        optional integer source cell IDs (cell bins)
    obsm["spatial"]  (n_obs, 2) DNB-grid coordinates, exact uint32 values
    uns["bin_type"]  "bins" or "cell_bins"; uns["bin_size"]: the square bin size
    uns["resolution"] physical pitch in nm per grid unit
    uns["sn"]        one-row DataFrame with the chip serial in column "sn"

``<r>`` in the filename is the clustering resolution, not the pitch. The table
keeps its values, row and feature order, annotations, embeddings and ``raw``;
when the matching raw GEF population is selected, every analysis feature,
coordinate and cell ID must belong to it. ``obs`` and ``var`` column names and
top-level ``uns`` keys that SpatialData rejects are sanitized with one warning
per frame or mapping; the names the reader reads or adds keep their spelling,
and ``uns`` values, their order and nested keys stay as stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd
from spatialdata.models import TableModel

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers._utils.names import sanitize_column_names, sanitize_uns_keys
from spatialdata_io.readers.stereoseq import _schemas

if TYPE_CHECKING:
    from collections.abc import Mapping
    from decimal import Decimal

    import numpy.typing as npt
    from anndata import AnnData

    from spatialdata_io.readers.stereoseq._cells import CellCounts
    from spatialdata_io.readers.stereoseq._layout import AnalysisSource
    from spatialdata_io.readers.stereoseq._square import SquareCounts

_MAX_EXPANDED_RESOLUTION_DIGITS = 64
# Columns and uns keys the reader relies on by name or adds before linkage.
_RESERVED_OBS = ("x", "y", "id", "region", "instance_id")
_RESERVED_UNS = ("bin_type", "bin_size", "resolution", "sn", POSITIONS_KEY, TableModel.ATTRS_KEY)


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Keep the owned analysis table separate from its independent point target."""

    table: AnnData
    coordinates: npt.NDArray[np.int64]
    instance_ids: pd.Index
    name: str
    population: str
    resolution: int


def read_analyses(
    sources: tuple[AnalysisSource, ...], *, counts: Mapping[str, SquareCounts | CellCounts], resolution: int | None
) -> tuple[tuple[AnalysisResult, ...], int | None]:
    """Read independent analyses and reconcile their normalized names and pitch.

    Returns the analyses and the common pitch in nm, starting from the raw
    products' ``resolution`` when known.
    """
    results: list[AnalysisResult] = []
    names: set[str] = set()
    for source in sources:
        analysis = read_analysis(source, counts=counts)
        if resolution is not None and analysis.resolution != resolution:
            raise _schemas.schema_error(
                source.file,
                "coordinate frame",
                found=f"analysis pitch {analysis.resolution} nm",
                expected=f"common grid pitch {resolution} nm",
            )
        resolution = analysis.resolution
        if analysis.name in names:
            raise _schemas.schema_error(
                source.file,
                "analysis",
                found=f"duplicate analysis name {analysis.name!r}",
                expected="distinct normalized analysis products",
            )
        names.add(analysis.name)
        results.append(analysis)
    return tuple(results), resolution


def read_analysis(source: AnalysisSource, *, counts: Mapping[str, SquareCounts | CellCounts]) -> AnalysisResult:
    """Read a native analysis product, preserving its values and row/feature order."""
    try:
        table = ad.read_h5ad(source.file)
    except OSError as error:
        add_reader_context(error, context=ReaderErrorContext("stereoseq", "analysis table"), path=source.file)
        raise
    schema = _schemas.validate_analysis_table(table, source.file)

    # validate_analysis_table proved that read_h5ad returned eager DataFrames.
    obs, var = cast("pd.DataFrame", table.obs), cast("pd.DataFrame", table.var)
    obs.columns = sanitize_column_names(
        list(obs.columns), source=f"stereoseq: the obs of {source.file.name}", reserved=_RESERVED_OBS
    )
    var.columns = sanitize_column_names(list(var.columns), source=f"stereoseq: the var of {source.file.name}")
    sanitize_uns_keys(table, source=f"stereoseq: the uns of {source.file.name}", reserved=_RESERVED_UNS)

    if schema.sample_id is not None and schema.sample_id != source.sample_id:
        raise _schemas.schema_error(
            source.file, "sample identity", found=schema.sample_id, expected=f"filename sample {source.sample_id!r}"
        )

    population_name = _population_name(source, schema)
    matching = counts.get(population_name)
    physical_resolution = schema.resolution
    if matching is not None:
        _match_counts(table, schema, matching, source)
        if physical_resolution is not None and physical_resolution != matching.resolution:
            raise _schemas.schema_error(
                source.file,
                "physical pitch",
                found=str(physical_resolution),
                expected=f"matching GEF pitch {matching.resolution} nm",
            )
        physical_resolution = matching.resolution

    if physical_resolution is None:
        raise _schemas.schema_error(
            source.file,
            "coordinate units",
            found="missing physical resolution",
            expected="native uns['resolution'] pitch in nm or matching GEF",
        )

    analysis_name = f"{population_name}_analysis"
    if source.resolution is not None:
        clustering_resolution = _resolution_name(source.resolution)
        analysis_name = f"{analysis_name}_r{clustering_resolution}"

    # Coordinates and names stay in native H5AD row order; matching only proves
    # membership and never replaces or reorders the imported analysis table.
    return AnalysisResult(
        table=table,
        coordinates=schema.coordinates,
        instance_ids=table.obs_names.copy(),
        name=analysis_name,
        population=population_name,
        resolution=physical_resolution,
    )


def _resolution_name(value: Decimal) -> str:
    """Retain exact decimal identity without context rounding or exponent expansion."""
    parts = value.as_tuple()
    digits = list(parts.digits)
    # The layout filename grammar produces finite Decimals, never sentinel exponents.
    exponent = cast("int", parts.exponent)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exponent += 1
    text = "".join(str(digit) for digit in digits)
    point = len(text) + exponent
    if 0 < point <= _MAX_EXPANDED_RESOLUTION_DIGITS and len(text) - point <= _MAX_EXPANDED_RESOLUTION_DIGITS:
        return text[:point] + ("_" + text[point:] if point < len(text) else "0" * (point - len(text)))
    if -_MAX_EXPANDED_RESOLUTION_DIGITS <= point <= 0:
        return "0_" + "0" * (-point) + text
    coefficient = text[0] + ("_" + text[1:] if len(text) > 1 else "")
    return f"{coefficient}e{point - 1:d}"


def _population_name(source: AnalysisSource, schema: _schemas.AnalysisSchema) -> str:
    """Reconcile filename evidence with native population metadata once."""
    population = source.population
    if population is None:
        if schema.bin_type != "bins" or schema.bin_size is None:
            raise _schemas.schema_error(
                source.file,
                "analysis population",
                found="missing historical square-bin identity",
                expected="native bin_type='bins' and bin_size",
            )
        population = schema.bin_size
    if isinstance(population, int):
        if schema.bin_type == "cell_bins" or (schema.bin_size is not None and schema.bin_size != population):
            raise _schemas.schema_error(
                source.file, "analysis population", found=repr(schema), expected=f"square bin {population}"
            )
        population_name = f"bin{population}"
    else:
        if schema.bin_type == "bins":
            raise _schemas.schema_error(
                source.file, "analysis population", found="square-bin metadata", expected="native cell-bin metadata"
            )
        population_name = population
    return population_name


def _match_counts(
    table: AnnData, schema: _schemas.AnalysisSchema, matching: SquareCounts | CellCounts, source: AnalysisSource
) -> None:
    """Compare filtered/reordered native identities against independent raw targets."""
    coordinates = matching.coordinates
    if isinstance(source.population, str) and schema.cell_ids is not None:
        # Integer centroids may coincide. Explicit semantic cell IDs identify
        # filtered cells independently; coordinates then verify that association.
        positions = matching.table.obs_names.get_indexer(pd.Index(schema.cell_ids))
        if np.any(positions < 0) or not np.array_equal(coordinates[positions], schema.coordinates):
            raise _schemas.schema_error(
                source.file,
                "analysis cell IDs/coordinates",
                found=repr(schema.cell_ids),
                expected="GEF cell-ID membership with exact coordinate agreement",
            )
    else:
        raw_index = pd.MultiIndex.from_arrays((coordinates[:, 0], coordinates[:, 1]))
        if isinstance(source.population, str) and not raw_index.is_unique:
            raise _schemas.schema_error(
                source.file,
                "GEF coordinates",
                found="ambiguous source centroids without semantic analysis cell IDs",
                expected="unique coordinate identities for native analysis matching",
            )
        analysis_index = pd.MultiIndex.from_arrays((schema.coordinates[:, 0], schema.coordinates[:, 1]))
        if np.any(raw_index.get_indexer(analysis_index) < 0):
            raise _schemas.schema_error(
                source.file,
                "analysis coordinates",
                found="coordinates absent from selected GEF",
                expected="exact source-coordinate membership",
            )
    if np.any(matching.table.var_names.get_indexer(table.var_names) < 0):
        raise _schemas.schema_error(
            source.file,
            "analysis features",
            found="features absent from selected GEF",
            expected="source feature-identity membership",
        )
