"""CosMx Arrow types, semantic validation and printed placement precision.

Arrow parses numeric fields directly to their declared widths. Placement
coordinates retain their source spelling to derive printed rounding
intervals; other annotations remain typed throughout ingestion. Named fields
accept any order and additional columns, with unique nonempty names without NUL.

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
  transcript, and at least 1 in metadata. Both use uint64 without a floating
  or signed intermediate. Counts are uint64 as well.
- Arrow's typed numeric grammar applies: spaces and tabs are trimmed, integers
  accept hexadecimal and reject ``2.0``, ``2e0``, ``+1`` and separators.
  Floats round directly to float64; required coordinates must be finite.
- Placement coordinates are plain decimal text. Half a unit in their last
  printed decimal place is their rounding bound; it must be finite and positive.
- Metadata annotations named ``Area``, ``Area.um2``, ``AspectRatio``,
  ``Width``, ``Height``, ``Mean.*`` and ``Max.*`` are nullable float64, with
  empty fields and ``NA`` missing; every other annotation stays exact text.
- ``z`` is the vendor's integer slice number in arbitrary units and can be
  negative. Arrow reads its hexadecimal spellings above
  ``0x7fffffffffffffff`` as wrapped negative values, which cannot be detected.

The vendor-affiliated `flat-file comparison
<https://nanostring-biostats.github.io/CosMx-Analysis-Scratch-Space/posts/flat-file-exports/flat-files-compare.html>`_
documents how these columns differ between releases.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.names import sanitize_column_names
from spatialdata_io.readers._utils.points import categorize_features

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

FOV = "fov"
CELL_ID = "cell_ID"
CELL = "cell"
REGION_KEY = "region"
FRAME_KEY = "spatial_pixel_frame"
CENTERS = ("CenterX_local_px", "CenterY_local_px", "CenterX_global_px", "CenterY_global_px")
_READER_OBS_COLUMNS = (FOV, CELL_ID, REGION_KEY, FRAME_KEY)
_TRANSCRIPT_REQUIRED = pa.schema(
    [
        ("x_global_px", pa.float64()),
        ("y_global_px", pa.float64()),
        ("target", pa.string()),
        (FOV, pa.uint64()),
        (CELL_ID, pa.uint64()),
        ("x_local_px", pa.float64()),
        ("y_local_px", pa.float64()),
    ]
)
_TRANSCRIPT_OPTIONAL = pa.schema([("z", pa.int64()), ("CellComp", pa.string()), (CELL, pa.string())])
_TRANSCRIPT_OUTPUT_NAMES = {
    "x_global_px": "x",
    "y_global_px": "y",
    "x_local_px": "x_local",
    "y_local_px": "y_local",
    "z": "z_raw",
}

# Vendor morphology and stain-intensity fields; every other metadata field stays exact text.
_NUMERIC_METADATA = re.compile(r"Area|Area\.um2|AspectRatio|Width|Height|(?:Mean|Max)\..+")
_FOV_POSITION_PIXELS = (("x_px", "y_px"), ("x_global_px", "y_global_px"))
_FOV_POSITION_STAGE = (("X_mm", "Y_mm"), ("x_global_mm", "y_global_mm"))
_DECIMAL = r"[+-]?(?:\d+(?:\.(?P<fraction>\d*))?|\.(?P<leading>\d+))(?:[eE](?P<exponent>[+-]?\d+))?"
_MIN_HALF_POWER, _MAX_HALF_POWER = -323, 308
# Round each representable decimal half-unit once; halving a rounded power of
# ten can misround by one float64 step, including at subnormal widths.
_HALF_UNITS = np.array([float(f"5e{power - 1}") for power in range(_MIN_HALF_POWER, _MAX_HALF_POWER + 1)])


def schema_error(path: ArtifactFile, artifact: Artifact, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a source contract violation with its reader and artifact role."""
    return ReaderFormatError(context=ReaderErrorContext("cosmx", artifact), path=path, found=found, expected=expected)


def validate_label_ids(cell_ids: pd.Index, dtype: np.dtype[np.generic], path: ArtifactFile) -> None:
    """Require positive cell IDs to fit their FOV's inspected integer label dtype."""
    maximum = int(cell_ids.max()) if len(cell_ids) else 0
    limit = np.iinfo(dtype.str).max
    if maximum > limit:
        raise schema_error(
            path,
            LABELS,
            found=f"cell_ID {maximum} outside the {dtype} label range",
            expected=f"cell_ID values representable by the target labels (at most {limit})",
        )


def _validate_columns(columns: Sequence[str], path: ArtifactFile, artifact: Artifact) -> None:
    """Reject empty or NUL-containing names; ``csv_header`` already rejects repeated names."""
    if not columns or any(not name or "\x00" in name for name in columns):
        raise schema_error(path, artifact, found=f"header {list(columns)!r}", expected="nonempty field names")


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


def count_types(columns: Sequence[str], path: ArtifactFile) -> pa.Schema:
    """Declare exact unsigned keys and every count feature in source order."""
    features = count_features(columns, path)
    return pa.schema([(FOV, pa.uint64()), (CELL_ID, pa.uint64()), *((name, pa.uint64()) for name in features)])


def metadata_types(columns: Sequence[str], path: ArtifactFile) -> pa.Schema:
    """Declare numeric annotations and retain only placement coordinates as source text."""
    _require(columns, (FOV, CELL_ID, *CENTERS), path, METADATA)
    for name in (REGION_KEY, FRAME_KEY):
        if name in columns:
            raise schema_error(
                path, METADATA, found=f"a source {name!r} column", expected="no column the reader adds to obs"
            )
    return pa.schema(
        [
            (
                name,
                pa.uint64()
                if name in {FOV, CELL_ID}
                else pa.float64()
                if _NUMERIC_METADATA.fullmatch(name)
                else pa.string(),
            )
            for name in columns
        ]
    )


def cell_keys(fovs: pd.Index, identifiers: pd.Index, path: ArtifactFile, artifact: Artifact) -> pd.MultiIndex:
    """Validate the exact compound key of one count or metadata population."""
    keys = pd.MultiIndex.from_arrays([fovs, identifiers])
    if keys.has_duplicates:
        raise schema_error(path, artifact, found="duplicate (fov, cell_ID) keys", expected="one row per cell")
    return keys


def normalize_cell_metadata(records: pa.Table, path: ArtifactFile) -> tuple[pd.DataFrame, pd.MultiIndex]:
    """Validate cell identities and return typed observation annotations and their exact keys."""
    for name in (FOV, CELL_ID):
        require_positive(records[name], path, METADATA, name)
    numeric = [field.name for field in records.schema if pa.types.is_floating(field.type)]
    text = [field.name for field in records.schema if pa.types.is_string(field.type) and field.name not in CENTERS]
    for name in numeric:
        require_finite(records[name], path, METADATA, name, nullable=True)
    obs = records.select([FOV, CELL_ID, *numeric, *text]).to_pandas(types_mapper=pd.ArrowDtype)
    keys = cell_keys(pd.Index(obs[FOV].array), pd.Index(obs[CELL_ID].array), path, METADATA)

    if CELL in obs and (obs[CELL].eq("").any() or not obs[CELL].is_unique):
        raise schema_error(
            path, METADATA, found="missing or repeated vendor cell IDs", expected="unique nonempty 'cell' values"
        )
    # AtoMx can repeat the exact vendor ID. Independent spellings remain annotations.
    if "cell_id" in obs and CELL in obs and obs["cell_id"].equals(obs[CELL]):
        obs = obs.drop(columns="cell_id")
    reserved = (*_READER_OBS_COLUMNS, CELL) if CELL in obs else _READER_OBS_COLUMNS
    obs.columns = sanitize_column_names(
        list(obs.columns), source="cosmx: the cell metadata CSV header", reserved=reserved
    )
    return obs, keys


def fov_position_types(columns: Sequence[str], path: ArtifactFile) -> pa.Schema | None:
    """Return the FOV and pixel-position fields, or None for an evidenced stage pair."""
    _validate_columns(columns, path, POSITIONS)
    identifiers = [name for name in (FOV, "FOV") if name in columns]
    if len(identifiers) != 1:
        raise schema_error(
            path, POSITIONS, found=f"columns {list(columns)!r}", expected="exactly one 'fov' or 'FOV' field"
        )
    names = set(columns)
    pixels = [pair for pair in _FOV_POSITION_PIXELS if names.intersection(pair)]
    stage = [pair for pair in _FOV_POSITION_STAGE if names.intersection(pair)]
    for pair in (*pixels, *stage):
        if not names.issuperset(pair):
            raise schema_error(
                path, POSITIONS, found=f"incomplete position pair {pair!r}", expected="both x and y fields"
            )
    if len(pixels) > 1:
        raise schema_error(path, POSITIONS, found="two pixel-position conventions", expected="one x/y pixel pair")
    if len(stage) > 1:
        raise schema_error(path, POSITIONS, found="two stage-position conventions", expected="one x/y millimetre pair")
    if not pixels and not stage:
        raise schema_error(
            path,
            POSITIONS,
            found=f"columns {list(columns)!r} without recognized coordinates",
            expected="a complete pixel or millimetre x/y position pair",
        )
    return pa.schema([(identifiers[0], pa.uint64()), *((name, pa.string()) for name in pixels[0])]) if pixels else None


def transcript_types(columns: Sequence[str], path: ArtifactFile) -> pa.Schema:
    """Project typed event fields in normalized output order for both encodings."""
    _require(columns, _TRANSCRIPT_REQUIRED.names, path, TRANSCRIPTS)
    return pa.schema([*_TRANSCRIPT_REQUIRED, *(field for field in _TRANSCRIPT_OPTIONAL if field.name in columns)])


def normalize_transcript_partition(frame: pd.DataFrame, *, features: pd.Index, path: ArtifactFile) -> pd.DataFrame:
    """Validate the projected Arrow partition and prepare its coordinates and annotations."""
    require_positive(pa.array(frame[FOV]), path, TRANSCRIPTS, FOV)
    for field in _TRANSCRIPT_REQUIRED:
        if pa.types.is_floating(field.type):
            require_finite(pa.array(frame[field.name]), path, TRANSCRIPTS, field.name)
    categories = categorize_features(frame["target"], features)
    if categories is None:
        raise schema_error(
            path, TRANSCRIPTS, found="missing or unknown targets", expected="targets from the count vocabulary"
        )
    frame["target"] = categories
    # PointsModel requires native NumPy x/y; local coordinates stay Arrow annotations.
    for name in ("x_global_px", "y_global_px"):
        frame[name] = frame[name].to_numpy(dtype=np.float64)
    return frame.rename(columns=_TRANSCRIPT_OUTPUT_NAMES)


def require_positive(column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str) -> None:
    """Reject missing or zero IDs after unsigned typed ingestion."""
    if column.null_count or pc.call_function("min", [column]).as_py() == 0:
        raise schema_error(
            path, artifact, found=f"missing or zero {name!r}", expected=f"positive uint64 {name!r} values"
        )


def require_finite(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str, *, nullable: bool = False
) -> None:
    """Validate finite physical values, permitting only declared optional nulls."""
    if (column.null_count and not nullable) or pc.call_function(
        "all", [pc.call_function("is_finite", [column])]
    ).as_py() is False:
        raise schema_error(path, artifact, found=f"missing or nonfinite {name!r}", expected=f"finite {name!r} values")


def placement_values(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return correctly rounded placement coordinates and their original printed half-units."""
    halves = half_units(column, path, artifact, name)
    values = pc.cast(column, pa.float64(), safe=True)
    require_finite(values, path, artifact, name)
    return values.to_numpy(), halves


def half_units(
    column: pa.Array | pa.ChunkedArray, path: ArtifactFile, artifact: Artifact, name: str
) -> npt.NDArray[np.float64]:
    """Return half a unit in the last printed decimal place of every source value.

    Each bound is rounded once to float64, independently of how many digits a
    given export happens to print.
    """
    parts = pc.call_function("extract_regex", [column], options=pc.ExtractRegexOptions(f"^{_DECIMAL}$"))
    if parts.null_count:
        raise schema_error(path, artifact, found=f"non-decimal {name!r} values", expected="plain decimal coordinates")

    fraction, leading, exponent_text = parts.flatten()
    fraction = (
        pc.call_function(
            "add", [pc.call_function("utf8_length", [fraction]), pc.call_function("utf8_length", [leading])]
        )
        .to_numpy()
        .astype(np.int64, copy=False)
    )
    # Optional groups are empty strings; Arrow's integer parser rejects '+'.
    exponent_text = pc.call_function(
        "if_else",
        [
            pc.call_function("equal", [exponent_text, ""]),
            "0",
            pc.call_function("utf8_ltrim", [exponent_text], options=pc.TrimOptions("+")),
        ],
    )
    expected = "finite positive float64 printed rounding half-widths"
    try:
        exponent = pc.cast(exponent_text, pa.int64(), safe=True).to_numpy()
    except pa.ArrowInvalid as error:
        raise schema_error(
            path, artifact, found=f"unrepresentable {name!r} rounding exponent", expected=expected
        ) from error

    # Compare before subtracting: a very negative exponent must not wrap to a
    # positive power. Powers -323 through 308 have finite positive half-units.
    if (exponent < fraction + _MIN_HALF_POWER).any():
        raise schema_error(
            path, artifact, found=f"{name!r} rounding half-widths that underflow to zero", expected=expected
        )
    power = exponent - fraction
    if (power > _MAX_HALF_POWER).any():
        raise schema_error(path, artifact, found=f"nonfinite {name!r} rounding half-widths", expected=expected)
    return _HALF_UNITS[power - _MIN_HALF_POWER]
