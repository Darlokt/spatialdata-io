"""Column and feature contracts of the Spatial Genomics ROI tables, transcripts and boundaries.

Every artifact is named ``<ROI>_<artifact>`` (see :mod:`._layout`). The tabular
artifacts are UTF-8 CSV files with one header record:

- ``CellxGene.csv``: the first field is the positional cell-ID column (its
  header is normally empty); every further, nonempty and unique header is a
  feature, in source order. Values are finite nonnegative integral counts;
  integral decimal and exponent spellings (``2.0``, ``3e0``) are accepted.
- ``CellCoordinates.csv``: named fields ``label`` (cell ID), ``area`` (µm²,
  nonnegative), ``center_x`` and ``center_y`` (cell centroid in µm), in any
  order; additional columns are ignored.
- ``TranscriptList.csv``: named fields ``name`` (a feature of the CellxGene
  header), ``x`` and ``y`` (µm), optional ``z`` (a nonnegative image-plane
  index, not a physical distance) and optional ``cell`` (the assigned cell ID,
  ``0`` when unassigned); additional columns are ignored.

``Boundaries.geojson`` is a nonempty GeoJSON ``FeatureCollection`` of Polygon or
MultiPolygon features in DAPI pixel coordinates. Each feature's top-level
``id`` (string or integer) is its cell ID; ``properties`` are independent
annotations, so ``properties.id`` need not match it.

Cell IDs are exact positive integers below ``2**64`` and are kept as int64, or
uint64 when a value needs it. Every artifact's header is checked for empty,
duplicate or NUL-containing names before Arrow or pandas could repair them.
Numbers are parsed from their text so that precision loss is detected:
coordinates must be finite float64 values without destructive underflow or
rounded integers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa

from spatialdata_io.readers._utils.csv import decode_float64, decode_integers
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy.typing as npt

    from spatialdata_io.readers._utils.paths import ArtifactFile

COUNTS_CONTEXT = ReaderErrorContext("seqfish", "CellxGene")
CELLS_CONTEXT = ReaderErrorContext("seqfish", "CellCoordinates")
TRANSCRIPTS_CONTEXT = ReaderErrorContext("seqfish", "TranscriptList")
BOUNDARIES_CONTEXT = ReaderErrorContext("seqfish", "Boundaries")

STRING_DTYPE = pd.ArrowDtype(pa.string())
FLOAT_DTYPE = pd.ArrowDtype(pa.float64())
CELL_COLUMNS = ("label", "area", "center_x", "center_y")


def _validate_columns(
    columns: Sequence[str], path: ArtifactFile, *, context: ReaderErrorContext, allow_first_empty: bool = False
) -> None:
    """Reject ambiguous headers before a storage parser can repair names."""
    invalid = not columns or len(set(columns)) != len(columns)
    invalid |= any(
        "\x00" in column or (not column and not (allow_first_empty and i == 0)) for i, column in enumerate(columns)
    )
    if invalid:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="empty, duplicate or NUL-containing names",
            expected="unique nonempty column names",
        )


def validate_count_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return the ordered count features that follow the positional cell-ID field."""
    _validate_columns(columns, path, context=COUNTS_CONTEXT, allow_first_empty=True)
    if not columns[1:]:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT,
            path=path,
            found="no feature columns",
            expected="a cell-ID field and at least one count feature",
        )
    return tuple(columns[1:])


def validate_cell_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return the projected cell fields, accepting any column order and extra annotations."""
    _validate_columns(columns, path, context=CELLS_CONTEXT)
    if not set(CELL_COLUMNS).issubset(columns):
        raise ReaderFormatError(
            context=CELLS_CONTEXT,
            path=path,
            found=f"columns {list(columns)!r}",
            expected=f"cell fields {CELL_COLUMNS!r}",
        )
    return CELL_COLUMNS


def validate_transcript_columns(columns: Sequence[str], path: ArtifactFile) -> tuple[str, ...]:
    """Return the projected transcript fields; additional vendor annotations are ignored."""
    _validate_columns(columns, path, context=TRANSCRIPTS_CONTEXT)
    if not {"name", "x", "y"}.issubset(columns):
        raise ReaderFormatError(
            context=TRANSCRIPTS_CONTEXT,
            path=path,
            found=f"columns {list(columns)!r}",
            expected="transcript fields name, x and y",
        )
    return ("name", "x", "y", *(column for column in ("z", "cell") if column in columns))


def normalize_transcript_partition(
    frame: pd.DataFrame, *, features: tuple[str, ...], path: ArtifactFile
) -> pd.DataFrame:
    """Validate one owned lazy partition without scanning any other source rows."""
    names = frame["name"]
    if names.isna().any() or not names.isin(features).all():
        raise ReaderFormatError(
            context=TRANSCRIPTS_CONTEXT,
            path=path,
            found="missing or unknown transcript names",
            expected="names in the expression vocabulary",
        )
    frame["name"] = pd.Categorical(names.to_numpy(), categories=features)
    for axis in ("x", "y"):
        frame[axis] = decode_float64(pa.array(frame[axis]), name=axis, context=TRANSCRIPTS_CONTEXT, path=path)
    if "cell" in frame:
        identifiers = decode_integers(
            pa.array(frame["cell"]), name="cell", minimum=0, context=TRANSCRIPTS_CONTEXT, path=path
        )
        frame["cell"] = pd.Series(identifiers, index=frame.index, dtype=pd.ArrowDtype(pa.uint64()))
    if "z" in frame:
        # Vendor z is a nonnegative image-plane index, not physical distance.
        planes = decode_integers(pa.array(frame.pop("z")), name="z", minimum=0, context=TRANSCRIPTS_CONTEXT, path=path)
        if planes.dtype.kind == "u":
            raise ReaderFormatError(
                context=TRANSCRIPTS_CONTEXT,
                path=path,
                found="z plane overflow",
                expected="nonnegative int64 image-plane indexes",
            )
        frame["z_raw"] = pd.Series(planes, index=frame.index, dtype=pd.ArrowDtype(pa.int64()))
    return frame


class BoundaryFeature(TypedDict):
    """Retain one source feature with authoritative top-level identity.

    ``geometry`` is validated and constructed by the shared GeoJSON polygon boundary.
    """

    type: Literal["Feature"]
    id: str | int
    geometry: object
    properties: dict[str, object] | None


def boundary_features(
    document: dict[str, object], path: ArtifactFile
) -> tuple[list[BoundaryFeature], npt.NDArray[np.int64 | np.uint64]]:
    """Validate the Feature schema and normalize the independent top-level feature IDs."""
    features = document.get("features")
    if document.get("type") != "FeatureCollection" or not isinstance(features, list) or not features:
        raise ReaderFormatError(
            context=BOUNDARIES_CONTEXT,
            path=path,
            found="empty or malformed boundary collection",
            expected="a nonempty FeatureCollection",
        )
    raw_ids: list[str | int] = []
    for feature in features:
        if not isinstance(feature, dict) or feature.get("type") != "Feature":
            raise ReaderFormatError(
                context=BOUNDARIES_CONTEXT,
                path=path,
                found="malformed boundary feature",
                expected="GeoJSON Features with explicit id",
            )
        identifier = feature.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, str | int):
            raise ReaderFormatError(
                context=BOUNDARIES_CONTEXT,
                path=path,
                found="missing or nonintegral feature id",
                expected="an explicit positive string or integer id",
            )
        raw_ids.append(identifier)
        # GeoJSON requires this field; null properties carry no annotations.
        properties = feature.get("properties")
        if properties is not None and not isinstance(properties, dict):
            raise ReaderFormatError(
                context=BOUNDARIES_CONTEXT,
                path=path,
                found="malformed feature properties",
                expected="an object or null properties",
            )
        if properties is not None and "geometry" in properties:
            raise ReaderFormatError(
                context=BOUNDARIES_CONTEXT,
                path=path,
                found="reserved properties.geometry",
                expected="geometry only in Feature.geometry",
            )
        feature["properties"] = properties
    identifiers = decode_integers(
        pa.array([str(value) for value in raw_ids], pa.string()),
        name="id",
        minimum=1,
        context=BOUNDARIES_CONTEXT,
        path=path,
    )
    if len(np.unique(identifiers)) != len(identifiers):
        raise ReaderFormatError(
            context=BOUNDARIES_CONTEXT,
            path=path,
            found="duplicate boundary IDs",
            expected="unique explicit boundary identifiers",
        )
    # The schema above validates feature identity and properties. Geometry
    # structure belongs to the shared GeoJSON polygon constructor.
    return cast("list[BoundaryFeature]", features), identifiers
