"""CosMx flat-file column contracts and exact lexical value decoding.

Every flat file is read as text, so exact integers, correctly rounded floats and
the printed precision of each coordinate come from the source spelling. Fields
are named: any order and additional fields are accepted, but names must be
unique, nonempty and free of NUL, so no parser can silently repair them.

Schemas (``<prefix>_<suffix>.csv`` or ``.csv.gz``)::

    exprMat_file         fov, cell_ID, then one count column per feature
                         (genes, negative probes, system controls) in source order
    metadata_file        fov, cell_ID, CenterX_local_px, CenterY_local_px,
                         CenterX_global_px, CenterY_global_px, any annotations
    fov_positions_file   fov, x_px, y_px                       (2021 release)
                         FOV, x_global_px, y_global_px, ...    (AtoMx with pixels)
                         FOV, X_mm, Y_mm, ...                  (millimetres only)
    tx_file              fov, cell_ID, x_local_px, y_local_px, x_global_px,
                         y_global_px, target, optional z, CellComp and cell

Values:

- ``fov`` is an integer of at least 1. ``cell_ID`` is an integer of at least 0
  in counts and transcripts, where 0 marks per-FOV background or an unassigned
  transcript, and at least 1 in metadata. Both become int64 without a floating
  intermediate.
- Counts are nonnegative integers.
- Coordinates are finite float64 decimals. Half a unit in the last printed
  decimal place of each value is its rounding bound, used for placement.
- Metadata annotations named ``Area``, ``Area.um2``, ``AspectRatio``,
  ``Width``, ``Height``, ``Mean.*`` and ``Max.*`` are nullable float64, with
  empty fields and ``NA`` missing; every other annotation stays exact text.
- ``z`` is the vendor's integer slice number in arbitrary units and can be
  negative.

The vendor-affiliated `flat-file comparison
<https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/flat-file-exports/flat-files-compare.html>`_
documents how these columns differ between releases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

type Artifact = Literal[
    "count matrix", "cell metadata", "FOV positions", "transcripts", "composite images", "cell labels"
]

COUNTS: Artifact = "count matrix"
METADATA: Artifact = "cell metadata"
POSITIONS: Artifact = "FOV positions"
TRANSCRIPTS: Artifact = "transcripts"
IMAGES: Artifact = "composite images"
LABELS: Artifact = "cell labels"

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
INTEGER_DTYPE = pd.ArrowDtype(pa.int64())

FOV = "fov"
CELL_ID = "cell_ID"
CELL = "cell"
REGION_KEY = "region"
FRAME_KEY = "spatial_pixel_frame"
CENTERS = ("CenterX_local_px", "CenterY_local_px", "CenterX_global_px", "CenterY_global_px")
TRANSCRIPT_REQUIRED = (FOV, CELL_ID, "x_local_px", "y_local_px", "x_global_px", "y_global_px", "target")
TRANSCRIPT_OPTIONAL = ("z", "CellComp", CELL)

# Vendor morphology and stain-intensity fields; every other metadata field stays exact text.
_NUMERIC_METADATA = re.compile(r"Area|Area\.um2|AspectRatio|Width|Height|(?:Mean|Max)\..+")
_FOV_POSITION_PIXELS = (("x_px", "y_px"), ("x_global_px", "y_global_px"))
_LITERAL_ZERO = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"
_DECIMAL = r"[+-]?(?:\d+(?:\.(?P<fraction>\d*))?|\.(?P<leading>\d+))(?:[eE](?P<exponent>[+-]?\d+))?"


def schema_error(path: ArtifactFile, artifact: Artifact, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source contract violation with its reader and artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("cosmx", artifact), path=path, found=found, expected=expected)


def validate_label_ids(cell_ids: npt.NDArray[np.int64], dtype: np.dtype[np.generic], path: ArtifactFile) -> None:
    """Require positive cell IDs to fit their FOV's inspected integer label dtype."""
    maximum = int(cell_ids.max(initial=0))
    limit = np.iinfo(dtype.str).max
    if maximum > limit:
        raise schema_error(
            path,
            LABELS,
            found=f"cell_ID {maximum} outside the {dtype} label range",
            expected=f"cell_ID values representable by the target labels (at most {limit})",
        )


def _validate_columns(columns: Sequence[str], path: ArtifactFile, artifact: Artifact) -> None:
    """Reject empty, duplicate, or NUL-containing names before parser repair."""
    if not columns or len(set(columns)) != len(columns) or any(not name or "\x00" in name for name in columns):
        raise schema_error(path, artifact, found=f"header {list(columns)!r}", expected="unique nonempty field names")


def _require(columns: Sequence[str], required: Sequence[str], path: ArtifactFile, artifact: Artifact) -> None:
    """Require named fields, accepting any order and additional fields."""
    _validate_columns(columns, path, artifact)
    missing = [name for name in required if name not in columns]
    if missing:
        raise schema_error(path, artifact, found=f"missing columns {missing!r}", expected=f"columns {list(required)!r}")


def count_features(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Declare every field except the two identifiers as expression, in source order."""
    _require(columns, (FOV, CELL_ID), path, COUNTS)
    features = tuple(name for name in columns if name not in {FOV, CELL_ID})
    if not features:
        raise schema_error(path, COUNTS, found="no count features", expected="at least one target column")
    return features


@dataclass(frozen=True, slots=True)
class MetadataColumns:
    """Partition cell metadata by declared semantic type, in source order."""

    names: tuple[str, ...]
    numeric: tuple[str, ...]
    text: tuple[str, ...]


def metadata_columns(columns: Sequence[str], path: ArtifactFile) -> MetadataColumns:
    """Type vendor morphology and intensities; keep every other annotation as text."""
    _require(columns, (FOV, CELL_ID, *CENTERS), path, METADATA)
    for name in (REGION_KEY, FRAME_KEY):
        if name in columns:
            raise schema_error(
                path, METADATA, found=f"a source {name!r} column", expected="no column the reader adds to obs"
            )
    annotations = [name for name in columns if name not in {FOV, CELL_ID, *CENTERS}]
    numeric = tuple(name for name in annotations if _NUMERIC_METADATA.fullmatch(name))
    return MetadataColumns(tuple(columns), numeric, tuple(name for name in annotations if name not in numeric))


def fov_position_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, str, str] | None:
    """Return the FOV and pixel-position fields, or None for millimetre-only exports."""
    _validate_columns(columns, path, POSITIONS)
    identifiers = [name for name in (FOV, "FOV") if name in columns]
    if len(identifiers) != 1:
        raise schema_error(
            path, POSITIONS, found=f"columns {list(columns)!r}", expected="exactly one 'fov' or 'FOV' field"
        )
    pixels = [pair for pair in _FOV_POSITION_PIXELS if set(pair).issubset(columns)]
    if len(pixels) > 1:
        raise schema_error(path, POSITIONS, found="two pixel-position conventions", expected="one x/y pixel pair")
    return (identifiers[0], *pixels[0]) if pixels else None


def transcript_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Project required event fields and the supported optional annotations."""
    _require(columns, TRANSCRIPT_REQUIRED, path, TRANSCRIPTS)
    return (*TRANSCRIPT_REQUIRED, *(name for name in TRANSCRIPT_OPTIONAL if name in columns))


def _lexical(column: pa.Array | pa.ChunkedArray) -> pd.Series:
    """View Arrow source text through pandas string methods without copying it."""
    return pd.Series(pd.arrays.ArrowExtensionArray(column), dtype=STRING_DTYPE)


def integer_values(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str, *, minimum: int | None
) -> npt.NDArray[np.int64]:
    """Decode exact int64 values without a floating intermediate, optionally bounded below."""
    expected = f"integer {name!r} values" + ("" if minimum is None else f" of at least {minimum}")
    if column.null_count:
        raise schema_error(path, artifact, found=f"missing {name!r} values", expected=expected)
    try:
        values = pc.cast(column, pa.int64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise schema_error(path, artifact, found=f"non-integer {name!r} values", expected=expected) from error
    if minimum is not None and (values < minimum).any():
        raise schema_error(path, artifact, found=f"{name!r} values below {minimum}", expected=expected)
    return values


def count_block(columns: Sequence[pa.Array], names: Sequence[str], path: ArtifactFile) -> npt.NDArray[np.int64]:
    """Decode a bounded rows-by-features count block with one vectorized pass.

    Only an invalid block is rescanned column by column, so the error names its
    feature.
    """
    try:
        values = integer_values(pa.concat_arrays(list(columns)), path, COUNTS, "count", minimum=0)
    except ReaderFormatError:
        for column, name in zip(columns, names, strict=True):
            integer_values(column, path, COUNTS, name, minimum=0)
        raise
    # A transposed column-major view keeps the decoded buffer without a copy.
    return values.reshape(len(columns), -1).T


def float_values(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> npt.NDArray[np.float64]:
    """Decode correctly rounded finite float64 values without silent underflow."""
    expected = f"finite float64 {name!r} values"
    if column.null_count:
        raise schema_error(path, artifact, found=f"missing {name!r} values", expected=expected)
    try:
        values = pc.cast(column, pa.float64()).to_numpy(zero_copy_only=False)
    except pa.ArrowInvalid as error:
        raise schema_error(path, artifact, found=f"nonnumeric {name!r} values", expected=expected) from error
    if not np.isfinite(values).all():
        raise schema_error(path, artifact, found=f"nonfinite {name!r} values", expected=expected)
    zero = values == 0
    if zero.any() and not _lexical(column)[zero].str.fullmatch(_LITERAL_ZERO).all():
        raise schema_error(path, artifact, found=f"{name!r} values that underflow to zero", expected=expected)
    return values


def optional_float_values(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> pd.Series:
    """Decode an annotation, treating empty fields and vendor ``NA`` as missing."""
    present = (~_lexical(column).isin(("", "NA"))).to_numpy(dtype=np.bool_)
    decoded = np.full(len(present), np.nan)
    if present.any():
        decoded[present] = float_values(column.filter(pa.array(present)), path, artifact, name)
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(decoded, mask=~present)), dtype=FLOAT_DTYPE)


def half_units(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> npt.NDArray[np.float64]:
    """Return half a unit in the last printed decimal place of every source value.

    This is the exact rounding bound of each written coordinate, independent of
    how many digits a given export happens to print.
    """
    lexical = _lexical(column)
    if not lexical.str.fullmatch(_DECIMAL).all():
        raise schema_error(path, artifact, found=f"non-decimal {name!r} values", expected="plain decimal coordinates")
    # Unmatched optional groups are missing or empty depending on the string storage.
    parts = lexical.str.extract(f"^{_DECIMAL}$").fillna("")
    fraction = (parts["fraction"].str.len() + parts["leading"].str.len()).to_numpy(dtype=np.int64)
    expected = "finite positive float64 printed rounding half-widths"
    try:
        exponent = parts["exponent"].mask(parts["exponent"].eq(""), "0").astype(np.int64).to_numpy()
    except (ValueError, OverflowError) as error:
        raise schema_error(
            path, artifact, found=f"unrepresentable {name!r} rounding exponent", expected=expected
        ) from error
    # Compare before subtracting: a very negative exponent must not wrap to a
    # positive power. 10**-323 / 2 is the smallest nonzero float64 half-unit.
    if (exponent < fraction - 323).any():
        raise schema_error(
            path, artifact, found=f"{name!r} rounding half-widths that underflow to zero", expected=expected
        )
    with np.errstate(over="ignore", under="ignore"):
        widths = 0.5 * np.power(10.0, exponent - fraction)
    if not np.isfinite(widths).all():
        raise schema_error(path, artifact, found=f"nonfinite {name!r} rounding half-widths", expected=expected)
    return widths
