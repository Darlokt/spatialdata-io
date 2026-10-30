"""Named seqFISH fields, exact integer identities, and finite coordinates."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
CELL_COLUMNS = ("label", "area", "center_x", "center_y")
_LITERAL_ZERO = r"[+-]?(?:0+(?:\.0*)?|\.0+)(?:[eE][+-]?\d+)?"


def schema_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Identify a seqFISH artifact schema violation."""
    return ReaderFormatError(
        context=ReaderErrorContext("seqfish", path.name), path=path, found=found, expected=expected
    )


def validate_columns(columns: Sequence[str], path: ArtifactFile, *, allow_first_empty: bool = False) -> None:
    """Reject ambiguous headers before a storage parser can repair names."""
    invalid = not columns or len(set(columns)) != len(columns)
    invalid |= any(
        "\x00" in column or (not column and not (allow_first_empty and i == 0)) for i, column in enumerate(columns)
    )
    if invalid:
        raise schema_error(
            path, found="empty, duplicate or NUL-containing names", expected="unique nonempty column names"
        )


def validate_count_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Declare the first positional cell-ID field and ordered count features."""
    validate_columns(columns, path, allow_first_empty=True)
    if not columns[1:]:
        raise schema_error(path, found="no feature columns", expected="a cell-ID field and at least one count feature")
    return tuple(columns[1:])


def validate_cell_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Require named cell fields while accepting arbitrary order and extra annotations."""
    validate_columns(columns, path)
    if not set(CELL_COLUMNS).issubset(columns):
        raise schema_error(path, found=f"columns {list(columns)!r}", expected=f"cell fields {CELL_COLUMNS!r}")
    return CELL_COLUMNS


def normalize_ids(
    series: pd.Series, path: ArtifactFile, *, allow_zero: bool = False
) -> npt.NDArray[np.int64 | np.uint64]:
    """Normalize exact integral values into one signed or unsigned 64-bit array."""
    values: list[int] = []
    lower = 0 if allow_zero else 1
    # CSV source columns are lexical Arrow strings. Cast their common unsigned
    # decimal form directly; never let numeric inference choose a float buffer.
    if (
        (series.dtype == STRING_DTYPE or isinstance(series.dtype, pd.StringDtype))
        and len(series)
        and not series.isna().any()
        and series.str.fullmatch(r"[0-9]+").all()
    ):
        try:
            integers = series.astype(pd.ArrowDtype(pa.uint64()))
        except pa.ArrowInvalid:
            pass
        else:
            if integers.min() < lower:
                raise schema_error(path, found="zero identifier", expected="positive cell identifiers")
            dtype = np.uint64 if integers.max() >= 2**63 else np.int64
            return integers.to_numpy(dtype=dtype)
    for value in series:
        token = str(value)
        if isinstance(value, (bool, np.bool_)):
            raise schema_error(path, found=f"identifier {token!r}", expected="finite integral 64-bit identifiers")
        try:
            exact = Decimal(float(value)) if isinstance(value, (float, np.floating)) else Decimal(token)
        except (InvalidOperation, ValueError) as error:
            raise schema_error(
                path, found=f"identifier {token!r}", expected="finite integral 64-bit identifiers"
            ) from error
        if not exact.is_finite() or exact < lower or exact >= 2**64 or exact != exact.to_integral_value():
            raise schema_error(
                path, found=f"identifier {token!r}", expected=f"integral identifiers in [{lower}, 2**64)"
            )
        values.append(int(exact))
    return np.asarray(values, dtype=np.uint64 if values and max(values) >= 2**63 else np.int64)


def normalize_floats(series: pd.Series, path: ArtifactFile, column: str) -> pd.Series:
    """Decode correctly rounded float64 values, rejecting overflow and destructive underflow."""
    lexical = series.astype(STRING_DTYPE)
    expected = "finite float64 values without underflow or integer precision loss"
    try:
        normalized = lexical.astype(FLOAT_DTYPE)
    except (ValueError, TypeError, pa.ArrowInvalid) as error:
        raise schema_error(path, found=f"nonnumeric {column} values", expected=expected) from error
    values = normalized.to_numpy(dtype=np.float64, na_value=np.nan)
    if not np.isfinite(values).all():
        raise schema_error(path, found=f"nonfinite or missing {column} values", expected=expected)

    # Native Arrow decoding rounds ordinary decimal coordinates correctly.
    # Only integral precision loss and a nonzero lexeme decoded as zero need
    # slower exact decimal comparisons; literal zeros bypass that work.
    candidates = np.abs(values) >= 2**53
    zero = values == 0
    if zero.any():
        literal_zero = lexical.str.fullmatch(_LITERAL_ZERO).to_numpy(dtype=np.bool_)
        candidates |= zero & ~literal_zero
    for row in np.flatnonzero(candidates):
        exact = Decimal(lexical.iloc[row])
        number = float(values[row])
        if (number == 0 and exact != 0) or (exact == exact.to_integral_value() and exact != Decimal(number)):
            raise schema_error(
                path,
                found=f"{column} value {lexical.iloc[row]!r}",
                expected=expected,
            )
    return normalized


def validate_transcript_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Project named transcript fields; additional vendor annotations are ignored."""
    validate_columns(columns, path)
    if not {"name", "x", "y"}.issubset(columns):
        raise schema_error(path, found=f"columns {list(columns)!r}", expected="transcript fields name, x and y")
    return ("name", "x", "y", *(column for column in ("z", "cell") if column in columns))


class BoundaryFeature(TypedDict):
    """Retain one source feature with authoritative top-level identity.

    ``geometry`` is validated and constructed by the shared GeoJSON polygon boundary.
    """

    type: Literal["Feature"]
    id: str | int
    geometry: object
    properties: dict[str, object] | None


def _normalize_boundary_properties(properties: object, identifier: str | int, path: ArtifactFile) -> None:
    """Reject annotation collisions while preserving compatible source properties."""
    if properties is not None and not isinstance(properties, dict):
        raise schema_error(path, found="malformed feature properties", expected="an object or null properties")
    if properties is not None and "geometry" in properties:
        raise schema_error(path, found="reserved properties.geometry", expected="geometry only in Feature.geometry")
    if properties is not None and "id" in properties:
        property_id = properties["id"]
        if isinstance(property_id, bool) or not isinstance(property_id, str | int):
            raise schema_error(path, found="ambiguous properties.id", expected="properties.id matching Feature.id")
        pair = normalize_ids(pd.Series([str(identifier), str(property_id)], dtype=STRING_DTYPE), path)
        if pair[0] != pair[1]:
            raise schema_error(path, found="conflicting properties.id", expected="properties.id matching Feature.id")
        # An optional duplicate identity column must not undergo pandas'
        # mixed-number inference before the authoritative index is attached.
        properties["id"] = str(pair[0])


def boundary_features(
    document: dict[str, object], path: ArtifactFile
) -> tuple[list[BoundaryFeature], npt.NDArray[np.int64 | np.uint64]]:
    """Validate the owned Feature schema and normalize independent feature IDs."""
    features = document.get("features")
    if document.get("type") != "FeatureCollection" or not isinstance(features, list) or not features:
        raise schema_error(
            path, found="empty or malformed boundary collection", expected="a nonempty FeatureCollection"
        )
    raw_ids: list[str | int] = []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise schema_error(path, found="malformed boundary feature", expected="GeoJSON Features with explicit id")
        identifier = feature.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, str | int):
            raise schema_error(
                path, found="missing or nonintegral feature id", expected="an explicit positive string or integer id"
            )
        raw_ids.append(identifier)
        # GeoJSON requires this field; null properties carry no annotations.
        feature["properties"] = feature.get("properties")
        _normalize_boundary_properties(feature["properties"], identifier, path)
    identifiers = normalize_ids(pd.Series([str(value) for value in raw_ids], dtype=STRING_DTYPE), path)
    if len(np.unique(identifiers)) != len(identifiers):
        raise schema_error(path, found="duplicate boundary IDs", expected="unique explicit boundary identifiers")
    # The schema above validates feature identity and properties. Geometry
    # structure belongs to the shared GeoJSON polygon constructor.
    return cast("list[BoundaryFeature]", features), identifiers


def normalize_transcript_partition(
    frame: pd.DataFrame, *, features: tuple[str, ...], path: ArtifactFile
) -> pd.DataFrame:
    """Validate one owned lazy partition without scanning any other source rows."""
    names = frame["name"]
    if names.isna().any() or not names.isin(features).all():
        raise schema_error(
            path, found="missing or unknown transcript names", expected="names in the expression vocabulary"
        )
    frame["name"] = pd.Categorical(names.to_numpy(), categories=features)
    for axis in ("x", "y"):
        frame[axis] = normalize_floats(frame[axis], path, axis).to_numpy(dtype=np.float64)
    if "cell" in frame:
        identifiers = normalize_ids(frame["cell"], path, allow_zero=True)
        frame["cell"] = pd.Series(
            identifiers, index=frame.index, dtype=pd.ArrowDtype(pa.from_numpy_dtype(identifiers.dtype))
        )
    if "z" in frame:
        # Vendor z is a nonnegative image-plane index, not physical distance.
        planes = normalize_ids(frame.pop("z"), path, allow_zero=True)
        if planes.dtype.kind == "u":
            raise schema_error(path, found="z plane overflow", expected="nonnegative int64 image-plane indexes")
        frame["z_raw"] = pd.Series(planes, index=frame.index, dtype=pd.ArrowDtype(pa.int64()))
    return frame
