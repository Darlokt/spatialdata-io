"""Import derived native AnnData products and validate source identity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import anndata as ad
import numpy as np
import pandas as pd

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


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Keep the owned analysis table separate from its independent point target."""

    table: AnnData
    coordinates: npt.NDArray[np.int64]
    instance_ids: pd.Index
    name: str
    population: str
    sample_id: str
    origin: tuple[int, int] | None
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
            raise _schemas.format_error(
                source.file,
                "coordinate frame",
                f"analysis pitch {analysis.resolution} nm",
                f"common grid pitch {resolution} nm",
            )
        resolution = analysis.resolution
        if analysis.name in names:
            raise _schemas.format_error(
                source.file,
                "analysis",
                f"duplicate analysis name {analysis.name!r}",
                "distinct normalized analysis products",
            )
        names.add(analysis.name)
        results.append(analysis)
    return tuple(results), resolution


def read_analysis(source: AnalysisSource, *, counts: Mapping[str, SquareCounts | CellCounts]) -> AnalysisResult:
    """Read a native analysis product, preserving its values and row/feature order."""
    table = ad.read_h5ad(source.file)
    schema = _schemas.validate_analysis_table(table, source.file)

    if schema.sample_id is not None and schema.sample_id != source.sample_id:
        raise _schemas.format_error(
            source.file, "sample identity", schema.sample_id, f"filename sample {source.sample_id!r}"
        )

    population_name = _population_name(source, schema)
    matching = counts.get(population_name)
    origin: tuple[int, int] | None = None
    physical_resolution = schema.resolution
    if matching is not None:
        _match_counts(table, schema, matching, source)
        if physical_resolution is not None and physical_resolution != matching.resolution:
            raise _schemas.format_error(
                source.file,
                "physical pitch",
                str(physical_resolution),
                f"matching GEF pitch {matching.resolution} nm",
            )
        physical_resolution = matching.resolution
        origin = matching.origin

    if physical_resolution is None:
        raise _schemas.format_error(
            source.file,
            "coordinate units",
            "missing physical resolution",
            "native uns['resolution'] pitch in nm or matching GEF",
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
        sample_id=source.sample_id,
        origin=origin,
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
            raise _schemas.format_error(
                source.file,
                "analysis population",
                "missing historical square-bin identity",
                "native bin_type='bins' and bin_size",
            )
        population = schema.bin_size
    if isinstance(population, int):
        if schema.bin_type == "cell_bins" or (schema.bin_size is not None and schema.bin_size != population):
            raise _schemas.format_error(source.file, "analysis population", repr(schema), f"square bin {population}")
        population_name = f"bin{population}"
    else:
        if schema.bin_type == "bins":
            raise _schemas.format_error(
                source.file, "analysis population", "square-bin metadata", "native cell-bin metadata"
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
            raise _schemas.format_error(
                source.file,
                "analysis cell IDs/coordinates",
                repr(schema.cell_ids),
                "GEF cell-ID membership with exact coordinate agreement",
            )
    else:
        raw_index = pd.MultiIndex.from_arrays((coordinates[:, 0], coordinates[:, 1]))
        if isinstance(source.population, str) and not raw_index.is_unique:
            raise _schemas.format_error(
                source.file,
                "GEF coordinates",
                "ambiguous source centroids without semantic analysis cell IDs",
                "unique coordinate identities for native analysis matching",
            )
        analysis_index = pd.MultiIndex.from_arrays((schema.coordinates[:, 0], schema.coordinates[:, 1]))
        if np.any(raw_index.get_indexer(analysis_index) < 0):
            raise _schemas.format_error(
                source.file,
                "analysis coordinates",
                "coordinates absent from selected GEF",
                "exact source-coordinate membership",
            )
    if np.any(matching.table.var_names.get_indexer(table.var_names) < 0):
        raise _schemas.format_error(
            source.file, "analysis features", "features absent from selected GEF", "source feature-identity membership"
        )
