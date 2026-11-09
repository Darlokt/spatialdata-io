"""Column contracts of the MERSCOPE count, metadata, boundary and transcript artifacts.

Instrument output and VPT output share these schemas. CSV files are decoded as
text and every field is normalized here, so integers never pass through an
inferred float and coordinates are rounded once from their source spelling.

``cell_by_gene.csv``
    A positional cell-identity column, unnamed or named (``cell``), followed
    by one column per feature in source order. Values are nonnegative
    integers, possibly spelled ``2.0`` or ``3e0``. Features named
    ``Blank-<positive integer>`` are negative controls.

``cell_metadata.csv``
    Named columns in any order: ``EntityID``, ``center_x`` and ``center_y``
    (micrometres) are required. Known physical fields (``volume``, bounds,
    shape descriptors) become float64, ``transcript_count`` an integer and
    ``fov`` a nullable int64 (merged entities have no FOV). Other fields stay
    text.

``cell_boundaries.parquet``, ``{cellpose,watershed}_micron_space.parquet``
    One row per cell and z slice: ``EntityID`` (integer or text), ``ZIndex``
    (nullable stack index) and ``Geometry`` (WKB Polygon or MultiPolygon in
    micrometres), plus an optional ``Type`` that must be ``Cell``. Other
    columns are not read.

``detected_transcripts.csv``
    ``gene``, ``global_x`` and ``global_y`` (micrometres) are required;
    ``global_z`` (a stack index, not a physical depth), ``cell_id`` (``-1``
    for unassigned), ``fov``, ``barcode_id`` and ``transcript_id`` are kept
    when present. Local pixel ``x``/``y`` and the unnamed row index are not
    read.

Cell identities are kept as exact text in every product, so integer IDs above
2**53 survive. Vendor schemas:
https://vizgen.github.io/vizgen-postprocessing/output_data_formats/entity_metadata_format.html,
https://vizgen.github.io/vizgen-postprocessing/output_data_formats/geometry_file_format.html and
https://vizgen.github.io/vizgen-postprocessing/output_data_formats/detected_transcripts_format.html.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.csv import decode_float64, decode_integers
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import numpy as np
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

type Artifact = Literal["count matrix", "cell metadata", "cell boundaries", "transcripts"]

COUNTS: Artifact = "count matrix"
METADATA: Artifact = "cell metadata"
BOUNDARIES: Artifact = "cell boundaries"
TRANSCRIPTS: Artifact = "transcripts"

STRING_DTYPE = pd.ArrowDtype(pa.string())
GEOMETRY_REPRESENTATION_DTYPE = pd.CategoricalDtype(["union_footprint", "source_components"])
_FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
_INTEGER_DTYPE = pd.ArrowDtype(pa.int64())
_BLANK_NAME = re.compile(r"Blank-[1-9][0-9]*")
_METADATA_FLOATS = frozenset(
    {
        "center_x",
        "center_y",
        "volume",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "anisotropy",
        "perimeter_area_ratio",
        "solidity",
    }
)
_TRANSCRIPT_OPTIONAL = ("global_z", "cell_id", "fov", "barcode_id", "transcript_id")
OUTPUT_ANNOTATIONS = ("region", "slide", "dataset_id", "cells_region")


def schema_error(path: ArtifactFile, artifact: Artifact, *, found: str, expected: str) -> ReaderFormatError:
    """Describe one MERSCOPE artifact contract violation."""
    return ReaderFormatError(
        context=ReaderErrorContext("merscope", artifact), path=path, found=found, expected=expected
    )


def _validate_columns(
    columns: Sequence[str], path: ArtifactFile, artifact: Artifact, *, first_empty: bool = False
) -> None:
    """Reject duplicate names before a parser could repair them; allow an unnamed index column."""
    if (
        not columns
        or len(set(columns)) != len(columns)
        or any("\x00" in name or (not name and not (first_empty and i == 0)) for i, name in enumerate(columns))
    ):
        raise schema_error(
            path, artifact, found="empty, duplicate or NUL-containing names", expected="unique named fields"
        )


def count_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return the ordered features that follow the positional cell-identity column."""
    _validate_columns(columns, path, COUNTS, first_empty=True)
    if not columns[1:]:
        raise schema_error(path, COUNTS, found="no count features", expected="a cell index followed by count features")
    return tuple(columns[1:])


def blank_columns(features: Sequence[str]) -> tuple[bool, ...]:
    """Identify only canonical positive-numbered ``Blank-N`` control names."""
    return tuple(_BLANK_NAME.fullmatch(feature) is not None for feature in features)


def normalize_cell_ids(
    series: pd.Series, path: ArtifactFile, artifact: Artifact, *, allow_unassigned: bool = False
) -> pd.Series:
    """Return exact nonempty text identities, allowing the ``-1`` sentinel only for assignments."""
    if series.dtype.kind in "fb":
        raise schema_error(
            path, artifact, found="floating or boolean cell IDs", expected="exact text or integer cell identities"
        )
    identifiers = series.astype(STRING_DTYPE)
    if identifiers.isna().any() or identifiers.eq("").any() or identifiers.str.contains("\x00", regex=False).any():
        raise schema_error(
            path, artifact, found="missing or malformed cell IDs", expected="nonempty exact cell identities"
        )
    if not allow_unassigned and identifiers.eq("-1").any():
        raise schema_error(
            path, artifact, found="unassigned cell ID in a cell population", expected="cell identities excluding -1"
        )
    return identifiers


def integer_values(
    source: pd.Series | pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> npt.NDArray[np.int64 | np.uint64]:
    """Decode nonnegative integers without a floating intermediate; uint64 only above int64."""
    # Parquet plane indexes arrive as Arrow integers; CSV fields as text.
    text = pa.array(source) if isinstance(source, pd.Series) else pc.cast(source, pa.string())
    return decode_integers(text, name=name, minimum=0, context=ReaderErrorContext("merscope", artifact), path=path)


def _float_values(series: pd.Series, path: ArtifactFile, artifact: Artifact, name: str) -> npt.NDArray[np.float64]:
    """Decode float64 values, rejecting nonfinite values, underflow and integer precision loss."""
    return decode_float64(pa.array(series), name=name, context=ReaderErrorContext("merscope", artifact), path=path)


def metadata_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Require named identities and centroids and keep every additional annotation."""
    _validate_columns(columns, path, METADATA)
    if not {"EntityID", "center_x", "center_y"}.issubset(columns):
        raise schema_error(
            path, METADATA, found=f"columns {list(columns)!r}", expected="EntityID, center_x and center_y metadata"
        )
    return tuple(columns)


def normalize_metadata(frame: pd.DataFrame, path: ArtifactFile) -> pd.DataFrame:
    """Index owned annotations by exact identity and type the known physical fields."""
    identifiers = normalize_cell_ids(frame.pop("EntityID"), path, METADATA)
    if not identifiers.is_unique:
        raise schema_error(path, METADATA, found="duplicate metadata identities", expected="one metadata row per cell")
    frame.index = pd.Index(identifiers.to_numpy(), name=None)
    for column in frame.columns:
        if column in _METADATA_FLOATS:
            frame[column] = pd.Series(
                _float_values(frame[column], path, METADATA, column), index=frame.index, dtype=_FLOAT_DTYPE
            )
    if "volume" in frame and frame["volume"].lt(0).any():
        raise schema_error(path, METADATA, found="negative volume", expected="nonnegative cell volumes")
    if "transcript_count" in frame:
        values = integer_values(frame["transcript_count"], path, METADATA, "transcript_count")
        frame["transcript_count"] = pd.Series(
            values, index=frame.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(values.dtype))
        )
    if "fov" in frame:
        # Vendor metadata permits absent FOV assignments for merged entities.
        missing = frame["fov"].eq("")
        values = integer_values(frame.loc[~missing, "fov"], path, METADATA, "fov")
        if values.dtype.kind == "u":
            raise schema_error(path, METADATA, found="FOV overflow", expected="nullable nonnegative int64 FOV indexes")
        frame["fov"] = pd.Series(pd.NA, index=frame.index, dtype=_INTEGER_DTYPE)
        frame.loc[~missing, "fov"] = values
    return frame


def validate_output_annotations(
    frame: pd.DataFrame, expected: Mapping[str, str], *, context: ReaderErrorContext
) -> None:
    """Reject conflicting source annotations before reader-owned names are attached."""
    for column in OUTPUT_ANNOTATIONS:
        if column not in frame:
            continue
        value = expected.get(column)
        if value is None:
            raise ReaderFormatError(
                context=context,
                found=f"source reserved annotation {column!r} without a returned target",
                expected="no cells_region annotation on an unlinked table",
            )
        values = frame[column]
        if values.isna().any() or not values.eq(value).all():
            raise ReaderFormatError(
                context=context,
                found=f"conflicting source {column!r} values",
                expected=f"only the reader-owned value {value!r} for {column!r}",
            )


def boundary_columns(names: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Project the identity, plane and geometry fields and the optional entity type."""
    _validate_columns(names, path, BOUNDARIES)
    columns = ("EntityID", "ZIndex", "Geometry")
    if not set(columns).issubset(names):
        raise schema_error(
            path, BOUNDARIES, found=f"columns {list(names)!r}", expected="EntityID, ZIndex and Geometry fields"
        )
    return (*columns, "Type") if "Type" in names else columns


def normalize_boundary_ids(table: pa.Table, path: ArtifactFile) -> pd.Index:
    """Validate slice identities, planes and types, keeping repeated rows for the footprint union."""
    ids = table.column("EntityID")
    if not (pa.types.is_integer(ids.type) or pa.types.is_string(ids.type) or pa.types.is_large_string(ids.type)):
        raise schema_error(
            path, BOUNDARIES, found=f"EntityID dtype {ids.type}", expected="exact integer or text identities"
        )
    identifiers = normalize_cell_ids(pd.Series(ids, dtype=STRING_DTYPE), path, BOUNDARIES)
    # VPT permits absent plane annotations. XY footprint construction uses every
    # geometry row, independently of this optional 3D stack annotation.
    planes = integer_values(table.column("ZIndex").drop_null(), path, BOUNDARIES, "ZIndex")
    if planes.dtype.kind == "u":
        raise schema_error(path, BOUNDARIES, found="ZIndex overflow", expected="nonnegative int64 plane indexes")
    if "Type" in table.column_names:
        types = pd.Series(table.column("Type"), dtype=STRING_DTYPE)
        if types.isna().any() or not types.isin(("Cell", "cell")).all():
            raise schema_error(path, BOUNDARIES, found="unsupported entity types", expected="only Cell/cell boundaries")
    if identifiers.empty:
        raise schema_error(
            path, BOUNDARIES, found="no cell boundaries", expected="a nonempty cell footprint population"
        )
    return pd.Index(identifiers, name="EntityID")


def validate_transcript_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return the projected coordinate, feature and supported annotation fields."""
    _validate_columns(columns, path, TRANSCRIPTS, first_empty=True)
    if not {"gene", "global_x", "global_y"}.issubset(columns):
        raise schema_error(
            path,
            TRANSCRIPTS,
            found=f"columns {list(columns)!r}",
            expected="gene, global_x and global_y transcript fields",
        )
    return ("gene", "global_x", "global_y", *(column for column in _TRANSCRIPT_OPTIONAL if column in columns))


def normalize_transcript_partition(
    frame: pd.DataFrame,
    *,
    features: tuple[str, ...],
    path: ArtifactFile,
    assignment: Literal["current", "original"],
    cell_population: pd.Index | None = None,
) -> pd.DataFrame:
    """Validate one owned partition and name its assignments by their provenance.

    ``cell_population`` is the unique Arrow-string index of current cells;
    assigned events must belong to it. ``None`` skips the check.
    """
    genes = frame["gene"]
    if genes.isna().any() or not genes.isin(features).all():
        raise schema_error(
            path, TRANSCRIPTS, found="missing or unknown gene names", expected="the complete source feature vocabulary"
        )
    frame["gene"] = pd.Categorical(genes.to_numpy(), categories=features)
    for column, axis in (("global_x", "x"), ("global_y", "y")):
        frame[axis] = _float_values(frame.pop(column), path, TRANSCRIPTS, column)
    for column in ("global_z", "fov", "barcode_id"):
        if column in frame:
            values = integer_values(frame.pop(column), path, TRANSCRIPTS, column)
            if values.dtype.kind == "u":
                raise schema_error(
                    path, TRANSCRIPTS, found=f"{column} overflow", expected="nonnegative int64 source indexes"
                )
            frame["z_raw" if column == "global_z" else column] = pd.Series(
                values, index=frame.index, dtype=_INTEGER_DTYPE
            )
    if "transcript_id" in frame:
        values = frame["transcript_id"].astype(STRING_DTYPE)
        if values.isna().any() or values.eq("").any():
            raise schema_error(
                path, TRANSCRIPTS, found="missing transcript_id", expected="nonempty codebook transcript identifiers"
            )
        frame["transcript_id"] = values
    if "cell_id" in frame:
        identifiers = normalize_cell_ids(frame.pop("cell_id"), path, TRANSCRIPTS, allow_unassigned=True)
        if assignment == "current" and cell_population is not None:
            _validate_assignments(identifiers, cell_population, path)
        frame["original_cell_id" if assignment == "original" else "cell_id"] = identifiers
    return frame


def _validate_assignments(identifiers: pd.Series, cell_population: pd.Index, path: ArtifactFile) -> None:
    """Require every assigned event to name a cell of the current population."""
    # An Arrow-string population index builds its hash table once for all partitions.
    assigned = pd.Index(identifiers[identifiers.ne("-1")])
    if (cell_population.get_indexer(assigned) < 0).any():
        raise schema_error(
            path,
            TRANSCRIPTS,
            found="unknown assigned cells",
            expected="current cell IDs or the -1 unassigned sentinel",
        )
