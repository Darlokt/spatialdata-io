"""MCMICRO marker-panel and MCQuant quantification column contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

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
_LITERAL_ZERO = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source contract violation with its reader and artifact identity."""
    return ReaderFormatError(
        context=ReaderErrorContext("mcmicro", path.name), path=path, found=found, expected=expected
    )


def validate_columns(columns: Sequence[str], path: ArtifactFile) -> None:
    """Reject empty, duplicate, or NUL-containing names before parser repair."""
    if not columns or len(set(columns)) != len(columns) or any(not name or "\x00" in name for name in columns):
        raise schema_error(path, found=f"header {list(columns)!r}", expected="unique nonempty field names")


@dataclass(frozen=True, slots=True)
class MarkerPanel:
    """Own canonical MCQuant feature IDs and the complete marker annotations."""

    features: tuple[str, ...]
    var: pd.DataFrame


def canonical_markers(names: Sequence[str]) -> tuple[str, ...]:
    """Suffix every occurrence of a repeated marker exactly as MCQuant does."""
    counts = Counter(names)
    return tuple(f"{name}_{names[:i].count(name) + 1}" if counts[name] > 1 else name for i, name in enumerate(names))


def normalize_markers(frame: pd.DataFrame, path: ArtifactFile) -> MarkerPanel:
    """Validate lexical marker rows and derive collision-free feature IDs in panel order."""
    if MARKER_NAME not in frame.columns:
        raise schema_error(path, found=f"columns {frame.columns.tolist()!r}", expected=f"a {MARKER_NAME!r} column")
    if frame.empty:
        raise schema_error(path, found="no markers", expected="one row per registered image channel")

    names = frame[MARKER_NAME]
    if names.eq("").any() or names.str.contains("\x00", regex=False).any():
        raise schema_error(path, found="empty or NUL-containing marker names", expected="nonempty marker names")
    features = canonical_markers(names.tolist())
    if len(set(features)) != len(features):
        raise schema_error(
            path,
            found=f"MCQuant feature IDs {list(features)!r} with generated collisions",
            expected="unique IDs after suffixing repeated marker names",
        )

    # Every optional column is free-form module metadata; keep its exact text.
    var = frame.astype(STRING_DTYPE)
    var.index = pd.Index(features)
    return MarkerPanel(features, var)


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
    validate_columns(columns, path)
    missing = [name for name in (CELL_ID, *features, *CENTROIDS) if name not in columns]
    if missing:
        raise schema_error(
            path,
            found=f"missing columns {missing!r}",
            expected=f"{CELL_ID!r}, every canonical marker and both centroids",
        )
    if REGION_KEY in columns:
        raise schema_error(path, found=f"a source {REGION_KEY!r} column", expected="no reader-owned linkage column")
    feature_set = set(features)
    annotations = [name for name in columns if name != CELL_ID and name not in feature_set]
    return QuantificationColumns(
        tuple(columns),
        features,
        tuple(name for name in annotations if name in MORPHOLOGY),
        tuple(name for name in annotations if name not in MORPHOLOGY),
    )


def _lexical(column: pa.Array | pa.ChunkedArray) -> pd.Series:
    """View Arrow source text through pandas string methods without copying it."""
    return pd.Series(pd.arrays.ArrowExtensionArray(column), dtype=STRING_DTYPE)


def float_values(column: pa.Array | pa.ChunkedArray, path: ArtifactFile, name: str) -> npt.NDArray[np.float64]:
    """Decode correctly rounded finite float64 values without silent underflow."""
    expected = f"finite float64 {name!r} values"
    lexical = _lexical(column)
    if column.null_count:
        raise schema_error(path, found=f"missing {name!r} values", expected=expected)
    try:
        values = pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise schema_error(path, found=f"nonnumeric {name!r} values", expected=expected) from error
    if not np.isfinite(values).all():
        raise schema_error(path, found=f"nonfinite {name!r} values", expected=expected)

    # A nonzero decimal spelling that decodes to zero would silently change sparsity.
    zero = values == 0
    if zero.any() and not lexical[zero].str.fullmatch(_LITERAL_ZERO).all():
        raise schema_error(path, found=f"{name!r} values that underflow to zero", expected=expected)
    return values


def float_block(columns: Sequence[pa.Array], names: Sequence[str], path: ArtifactFile) -> npt.NDArray[np.float64]:
    """Decode a bounded rows-by-markers block with one vectorized pass over its text.

    The column-major text is decoded at once; only an invalid block is rescanned
    column by column, so the error names the offending marker.
    """
    try:
        values = float_values(pa.concat_arrays(list(columns)), path, "marker")
    except ReaderFormatError:
        for column, name in zip(columns, names, strict=True):
            float_values(column, path, name)
        raise
    # A transposed column-major view keeps the decoded buffer without a copy.
    return values.reshape(len(columns), -1).T


def optional_float_values(column: pa.Array | pa.ChunkedArray, path: ArtifactFile, name: str) -> pd.Series:
    """Decode a morphology annotation, treating only empty fields as missing."""
    present = _lexical(column).ne("").to_numpy(dtype=np.bool_)
    decoded = np.full(len(present), np.nan)
    if present.any():
        decoded[present] = float_values(column.filter(pa.array(present)), path, name)
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(decoded, mask=~present)), dtype=FLOAT_DTYPE)


def cell_ids(column: pa.Array | pa.ChunkedArray, label_dtype: np.dtype[np.generic], path: ArtifactFile) -> pd.Series:
    """Decode exact positive unique label IDs that fit the segmentation dtype."""
    expected = f"unique positive integer {CELL_ID!r} values representable as {label_dtype}"
    if column.null_count:
        raise schema_error(path, found=f"missing {CELL_ID!r} values", expected=expected)
    try:
        # Arrow parses decimal integers exactly; float spellings and overflow fail.
        identifiers = pc.cast(column, pa.int64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise schema_error(path, found=f"non-integer {CELL_ID!r} values", expected=expected) from error
    # Discovery admits only integer label dtypes; the type string selects their exact range.
    limit = min(np.iinfo(label_dtype.str).max, np.iinfo(np.int64).max)
    if (identifiers <= 0).any() or (identifiers > limit).any() or len(np.unique(identifiers)) != len(identifiers):
        raise schema_error(path, found=f"{CELL_ID!r} values outside the label contract", expected=expected)
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(identifiers)), dtype=CELL_ID_DTYPE)
