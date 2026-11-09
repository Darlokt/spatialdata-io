"""MCMICRO ``markers.csv`` and MCQuant quantification column contracts.

``markers.csv`` (one row per registered image channel, in channel order)::

    marker_name, cycle, channel_number, ...  # header; only marker_name is required
    DAPI, 1, 1, ...
    CD3, 1, 2, ...

Every column, including ``marker_name``, is kept as exact text. The canonical
feature ID of a marker is its name, or ``<name>_<k>`` (one-based occurrence)
for every occurrence of a name that repeats, exactly as MCQuant names its
columns; a generated ID that collides with another name fails.

MCQuant writes one ``<image>--<method>_<mask>.csv`` per label image
(https://github.com/labsyspharm/quantification/blob/61a1f539665864d7a2896c6754d90de404c16759/mcquant/SingleCellDataExtraction.py#L47-L76)::

    CellID,<marker IDs...>,X_centroid,Y_centroid,column_centroid,row_centroid,
    Area,MajorAxisLength,MinorAxisLength,Eccentricity,Solidity,Extent,Orientation,...

- ``CellID``: the label value of the cell, a positive integer.
- one column per canonical marker ID: mean intensity, the expression values.
- the regionprops morphology columns in ``MORPHOLOGY``, decoded as float64.
  ``X_centroid``/``Y_centroid`` (required) are ``centroid-1``/``centroid-0``,
  zero-based pixel-index coordinates of the label image.
- any other column (for example ``<marker>_intensity_max`` from extra
  intensity properties) is kept as exact text.

Columns are matched by name in any order. Names must be unique, nonempty and
free of NUL before a parser could repair them. The reader-owned ``region``
column may not occur, and no canonical marker ID may name ``CellID`` or a
centroid, so every column has one role.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.csv import decode_float64, decode_integers
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
CELL_ID_DTYPE = pd.ArrowDtype(pa.int64())

MARKER_NAME = "marker_name"
CELL_ID = "CellID"
REGION_KEY = "region"
CENTROIDS = ("X_centroid", "Y_centroid")
# MCQuant's regionprops columns at revision 61a1f539665864d7a2896c6754d90de404c16759.
MORPHOLOGY = frozenset(
    {
        *CENTROIDS,
        "column_centroid",
        "row_centroid",
        "Area",
        "MajorAxisLength",
        "MinorAxisLength",
        "Eccentricity",
        "Solidity",
        "Extent",
        "Orientation",
    }
)

MARKERS_CONTEXT = ReaderErrorContext("mcmicro", "markers")
QUANTIFICATION_CONTEXT = ReaderErrorContext("mcmicro", "quantification")


def validate_columns(columns: Sequence[str], path: ArtifactFile, *, context: ReaderErrorContext) -> None:
    """Reject empty, duplicate, or NUL-containing names before parser repair."""
    if not columns or len(set(columns)) != len(columns) or any(not name or "\x00" in name for name in columns):
        raise ReaderFormatError(
            context=context, path=path, found=f"header {list(columns)!r}", expected="unique nonempty field names"
        )


@dataclass(frozen=True, slots=True)
class MarkerPanel:
    """Own canonical MCQuant feature IDs and the complete marker annotations."""

    features: tuple[str, ...]
    var: pd.DataFrame


def canonical_markers(names: Sequence[str]) -> tuple[str, ...]:
    """Suffix every occurrence of a repeated marker exactly as MCQuant does."""
    counts = Counter(names)
    occurrences: Counter[str] = Counter()
    features: list[str] = []
    for name in names:
        occurrences[name] += 1
        features.append(f"{name}_{occurrences[name]}" if counts[name] > 1 else name)
    return tuple(features)


def normalize_markers(frame: pd.DataFrame, path: ArtifactFile) -> MarkerPanel:
    """Index the owned Arrow-string panel with collision-free MCQuant feature IDs."""
    if MARKER_NAME not in frame.columns:
        raise ReaderFormatError(
            context=MARKERS_CONTEXT,
            path=path,
            found=f"columns {frame.columns.tolist()!r}",
            expected=f"a {MARKER_NAME!r} column",
        )
    if frame.empty:
        raise ReaderFormatError(
            context=MARKERS_CONTEXT, path=path, found="no markers", expected="one row per registered image channel"
        )

    names = frame[MARKER_NAME]
    if names.eq("").any() or names.str.contains("\x00", regex=False).any():
        raise ReaderFormatError(
            context=MARKERS_CONTEXT,
            path=path,
            found="empty or NUL-containing marker names",
            expected="nonempty marker names",
        )
    features = canonical_markers(names.tolist())
    if len(set(features)) != len(features):
        raise ReaderFormatError(
            context=MARKERS_CONTEXT,
            path=path,
            found=f"MCQuant feature IDs {list(features)!r} with generated collisions",
            expected="unique IDs after suffixing repeated marker names",
        )

    # Every optional column is free-form module metadata; keep its exact text.
    frame.index = pd.Index(features)
    return MarkerPanel(features, frame)


@dataclass(frozen=True, slots=True)
class QuantificationColumns:
    """Partition one MCQuant header by declared role, in source order."""

    names: tuple[str, ...]
    features: tuple[str, ...]
    morphology: tuple[str, ...]
    text: tuple[str, ...]


def quantification_columns(
    columns: Sequence[str], features: tuple[str, ...], path: ArtifactFile
) -> QuantificationColumns:
    """Select expression by the marker panel, never by numeric appearance."""
    validate_columns(columns, path, context=QUANTIFICATION_CONTEXT)
    conflicts = sorted(set(features).intersection((CELL_ID, *CENTROIDS)))
    if conflicts:
        raise ReaderFormatError(
            context=QUANTIFICATION_CONTEXT,
            path=path,
            found=f"canonical marker IDs {conflicts!r} overlap required quantification fields",
            expected="marker IDs distinct from CellID and centroid fields",
        )
    missing = [name for name in (CELL_ID, *features, *CENTROIDS) if name not in columns]
    if missing:
        raise ReaderFormatError(
            context=QUANTIFICATION_CONTEXT,
            path=path,
            found=f"missing columns {missing!r}",
            expected=f"{CELL_ID!r}, every canonical marker and both centroids",
        )
    if REGION_KEY in columns:
        raise ReaderFormatError(
            context=QUANTIFICATION_CONTEXT,
            path=path,
            found=f"a source {REGION_KEY!r} column",
            expected="no reader-owned linkage column",
        )
    feature_set = set(features)
    annotations = [name for name in columns if name != CELL_ID and name not in feature_set]
    return QuantificationColumns(
        tuple(columns),
        features,
        tuple(name for name in annotations if name in MORPHOLOGY),
        tuple(name for name in annotations if name not in MORPHOLOGY),
    )


def optional_float_values(column: pa.Array | pa.ChunkedArray, path: ArtifactFile, name: str) -> pd.Series:
    """Decode a morphology annotation, treating only empty fields as missing."""
    present = pd.Series(pd.arrays.ArrowExtensionArray(column), dtype=STRING_DTYPE).ne("").to_numpy(dtype=np.bool_)
    decoded = np.full(len(present), np.nan)
    if present.any():
        decoded[present] = decode_float64(
            column.filter(pa.array(present)), name=name, context=QUANTIFICATION_CONTEXT, path=path
        )
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(decoded, mask=~present)), dtype=FLOAT_DTYPE)


def cell_ids(column: pa.Array | pa.ChunkedArray, label_dtype: np.dtype[np.integer], path: ArtifactFile) -> pd.Series:
    """Decode exact positive label IDs that fit the segmentation dtype."""
    identifiers = decode_integers(column, name=CELL_ID, minimum=1, context=QUANTIFICATION_CONTEXT, path=path)
    limit = min(np.iinfo(label_dtype).max, np.iinfo(np.int64).max)
    if len(identifiers) and int(identifiers.max()) > limit:
        raise ReaderFormatError(
            context=QUANTIFICATION_CONTEXT,
            path=path,
            found=f"{CELL_ID!r} value {int(identifiers.max())} above {limit}",
            expected=f"positive integer {CELL_ID!r} values representable as {label_dtype}",
        )
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(identifiers, pa.int64())), dtype=CELL_ID_DTYPE)


def validate_position_columns(columns: QuantificationColumns, *, multiple: bool, path: ArtifactFile) -> None:
    """Reject imported fields that would collide with generated per-row frames."""
    if multiple and {"spatial_frame", "spatial_pixel_frame"}.intersection(columns.names):
        raise ReaderFormatError(
            context=QUANTIFICATION_CONTEXT,
            path=path,
            found="existing spatial_frame or spatial_pixel_frame column",
            expected="unoccupied generated position frame columns",
        )
