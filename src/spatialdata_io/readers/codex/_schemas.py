"""Validate the CODEX cell-table schema: named fields, cycle channels, IDs, regions and numbers.

A processor-style regional cell table (CSV or FCS) has one row per segmented
cell and these named fields, in any order::

    cell_id        integral cell identifier, unique within the region
    region         one region per table: a positive integer or a name
    x, y           cell position in native region pixels
    cycNNN_chNNN   one integrated intensity per cycle and channel (>= 1 field)
    ...            any further field (tile_num, z, size, tsne_x, ...) is metadata

Only complete ``cyc<digits>_ch<digits>`` names denote expression; a name such as
``cyc001_ch002_note`` is metadata. Compensated intensities may be negative.
Akoya CSV exports may qualify every header as ``short_name:long_label``; the
short names identify fields and the expression long labels become markers.

Cell IDs are uint64 (zero allowed) in both containers, so CSV and FCS tables
concatenate. CSV coordinates and expression are float64, and regions and
unknown annotations remain literal text. CSV numeric fields follow Arrow's
typed CSV grammar; expression and coordinates must be finite. FCS IDs are cast
exactly from their stored values; other FCS parameters retain their native
numeric widths. Processor output contract:
https://help.codex.bio/codex/processor/technical-notes/expected-output
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import validate_element_name

if TYPE_CHECKING:
    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
REQUIRED_COLUMNS = frozenset({"cell_id", "region", "x", "y"})
CELL_TABLE_CONTEXT = ReaderErrorContext("codex", "cell table")
_CYCLE_COLUMN = re.compile(r"cyc\d+_ch\d+")
# The region names the shapes element. Other element names are fixed, and
# SpatialData compares names case-insensitively across element types.
_RESERVED_REGIONS = frozenset({"table", "images"})
_CELL_ID_EXPECTED = "unique uint64 cell identifiers"
_REGION_EXPECTED = (
    "a positive integer region or a valid element name (letters, digits, '_-.') other than 'table' or 'images' "
    "in any letter case"
)


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a cell-table schema violation."""
    return ReaderFormatError(context=CELL_TABLE_CONTEXT, path=path, found=found, expected=expected)


def cell_columns(columns: list[str], path: ArtifactFile) -> tuple[list[str], list[str]]:
    """Validate source names and separate cycle features from metadata in source order."""
    if (
        not columns
        or any(not isinstance(column, str) or not column or "\x00" in column for column in columns)
        or len(set(columns)) != len(columns)
    ):
        raise schema_error(
            path, found="empty or duplicate column names", expected="nonempty unique source column names"
        )

    if not REQUIRED_COLUMNS.issubset(columns):
        raise schema_error(
            path, found=f"columns {columns!r}", expected=f"required columns {sorted(REQUIRED_COLUMNS)!r}"
        )

    features = [column for column in columns if _CYCLE_COLUMN.fullmatch(column)]
    if not features:
        raise schema_error(path, found="no cycle channels", expected="at least one cycNNN_chNNN expression column")

    feature_set = set(features)
    return features, [column for column in columns if column not in feature_set]


def csv_column_names(columns: list[str], path: ArtifactFile) -> tuple[list[str], list[str] | None]:
    """Normalize complete Akoya short-name:long-label headers in source order.

    Plain normalized tables retain additional metadata names literally. A
    qualified export must qualify every field; duplicate short names remain
    errors at the ordinary cell-column boundary. Split only the first colon.
    """
    if REQUIRED_COLUMNS.issubset(columns):
        return columns, None
    if not columns or not all(":" in column for column in columns):
        return columns, None
    names: list[str] = []
    labels: list[str] = []
    for column in columns:
        name, _, label = column.partition(":")
        if not label or "\x00" in label:
            raise schema_error(path, found="empty or NUL-containing long label", expected="nonempty Akoya long labels")
        names.append(name)
        labels.append(label)
    return names, labels


def fcs_cell_ids(values: pd.Series, path: ArtifactFile) -> pd.Series:
    """Cast native FCS cell IDs exactly to uint64; stored floats must be integral and nonnegative."""
    try:
        identifiers = pc.cast(pa.array(values), pa.uint64(), safe=True)
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, found=f"cell_id values that are not uint64 integers ({error})", expected=_CELL_ID_EXPECTED
        ) from error
    if identifiers.null_count:
        raise schema_error(path, found="missing cell_id", expected=_CELL_ID_EXPECTED)
    return pd.Series(identifiers, index=values.index, dtype=pd.ArrowDtype(pa.uint64()))


def region_name(regions: pd.Series, path: ArtifactFile) -> str:
    """Require one region and return it as the shapes element name.

    Positive integral values become decimal strings; names must be valid
    SpatialData element names that do not collide with ``table`` or ``images``.
    """
    if regions.isna().any() or regions.ne(regions.iloc[0]).any():
        raise schema_error(path, found="missing or multiple regions", expected="one non-null region per cell table")

    region = regions.iloc[0]
    if isinstance(region, (int, float, np.integer, np.floating)):
        if isinstance(region, bool) or not np.isfinite(region) or region <= 0 or region != int(region):
            raise schema_error(path, found=f"region {region!r}", expected=_REGION_EXPECTED)
        return str(int(region))

    try:
        return validate_element_name(region, parameter="region", reserved=_RESERVED_REGIONS)
    except (TypeError, ValueError) as error:
        raise schema_error(path, found=f"region {region!r}", expected=_REGION_EXPECTED) from error


def csv_types(columns: list[str], features: list[str]) -> pa.Schema:
    """Declare fixed CSV types after source names and cycle roles were validated."""
    numeric = {*features, "x", "y"}
    return pa.schema(
        [
            (name, pa.uint64() if name == "cell_id" else pa.float64() if name in numeric else pa.string())
            for name in columns
        ]
    )


def validate_csv_annotations(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Reject NUL in source text; typed CSV ingestion validates numeric fields and widths."""
    for column in frame:
        if column not in {"cell_id", "x", "y"} and frame[column].str.contains("\x00", regex=False).any():
            raise schema_error(
                path, found=f"NUL character in {column!r}", expected="text fields without NUL characters"
            )


def normalize_fcs_metadata(frame: pd.DataFrame, path: ArtifactFile) -> None:
    """Keep owned FCS annotations and primitive widths in explicit Arrow columns."""
    for column, dtype in frame.dtypes.items():
        if dtype.kind not in "iuf":
            raise schema_error(path, found=f"FCS parameter dtype {dtype}", expected="real numeric event parameters")

        frame[column] = frame[column].astype(pd.ArrowDtype(pa.from_numpy_dtype(dtype)))


def expression_dtype(frame: pd.DataFrame, features: list[str], path: ArtifactFile) -> np.dtype:
    """Choose a real kernel dtype for native FCS buffers or validated XY annotations."""
    dtypes: list[np.dtype] = []
    for column in features:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.ArrowDtype):
            dtype = np.dtype(dtype.pyarrow_dtype.to_pandas_dtype())

        if not isinstance(dtype, np.dtype) or dtype.kind not in "iuf":
            raise schema_error(
                path, found="nonnumeric cycle values or x/y coordinates", expected="finite real numeric values"
            )
        dtypes.append(dtype)

    return np.result_type(*dtypes)


def validate_expression(values: npt.NDArray[np.number], path: ArtifactFile) -> None:
    """Reject nonfinite cycle intensities before sparse construction."""
    if not np.isfinite(values).all():
        raise schema_error(path, found="nonfinite cycle intensities", expected="finite real cycle intensities")


def validate_observations(observations: pd.DataFrame, path: ArtifactFile) -> npt.NDArray[np.number]:
    """Validate normalized uint64 cell identity and finite region-pixel coordinates."""
    if not observations["cell_id"].is_unique:
        raise schema_error(path, found="duplicate cell_id values", expected=_CELL_ID_EXPECTED)

    coordinates = observations[["x", "y"]].to_numpy(dtype=expression_dtype(observations, ["x", "y"], path))
    if not np.isfinite(coordinates).all():
        raise schema_error(path, found="invalid x/y coordinates", expected="finite real region pixel coordinates")

    return coordinates
