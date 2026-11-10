"""Arrow types and field contracts of the Spatial Genomics ROI tables, transcripts and boundaries.

Every artifact is named ``<ROI>_<artifact>`` (see :mod:`._layout`). The tabular
artifacts are UTF-8 CSV files with one header record, read by Arrow with the
types declared here:

- ``CellxGene.csv``: the first field is the positional cell ID (its header is
  normally empty); every further, nonempty header is a feature, in source
  order, holding a count. Cell IDs are positive ``uint64``, counts ``uint64``.
- ``CellCoordinates.csv``: named fields ``label`` (cell ID, positive
  ``uint64``), ``area`` (µm², finite and nonnegative ``float64``), ``center_x``
  and ``center_y`` (cell centroid in µm, finite ``float64``), in any order;
  additional columns are ignored.
- ``TranscriptList.csv``: named fields ``name`` (a feature of the CellxGene
  header), ``x`` and ``y`` (µm, finite ``float64``), optional ``z`` (a
  nonnegative ``uint64`` image-plane index, not a physical distance) and
  optional ``cell`` (the assigned cell ID as ``uint64``, ``0`` when
  unassigned); additional columns are ignored.

Read numeric fields cannot be empty; ignored annotations may be. Numbers follow
Arrow's CSV grammar: surrounding spaces are trimmed and integers may be hexadecimal (``0x10``), but ``2.0`` or ``+1``
is not an integer; floats are correctly rounded, so ``1e-400`` reads as 0.
Header names must be unique; CellxGene feature names must also be nonempty and
free of NUL characters, because they name the table's features.

``Boundaries.geojson`` is a nonempty GeoJSON ``FeatureCollection`` of Polygon or
MultiPolygon features in DAPI pixel coordinates. Each feature's top-level
``id`` (string or integer) is its cell ID, cast by Arrow to the cell-ID type;
``properties`` are independent annotations, so ``properties.id`` need not match
it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypedDict, cast

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from spatialdata_io.readers._utils.paths import ArtifactFile

COUNTS_CONTEXT = ReaderErrorContext("seqfish", "CellxGene")
CELLS_CONTEXT = ReaderErrorContext("seqfish", "CellCoordinates")
TRANSCRIPTS_CONTEXT = ReaderErrorContext("seqfish", "TranscriptList")
BOUNDARIES_CONTEXT = ReaderErrorContext("seqfish", "Boundaries")

CELL_ID = pa.uint64()
COUNT = pa.uint64()
CELL_TYPES = {"label": CELL_ID, "area": pa.float64(), "center_x": pa.float64(), "center_y": pa.float64()}
TRANSCRIPT_TYPES = {"name": pa.string(), "x": pa.float64(), "y": pa.float64(), "z": pa.uint64(), "cell": pa.uint64()}
_TRANSCRIPT_REQUIRED = ("name", "x", "y")


def count_types(header: Sequence[str], path: ArtifactFile) -> pa.Schema:
    """Return the Arrow types of a CellxGene header: the positional cell ID, then one count per feature."""
    features = header[1:]
    if not features:
        raise ReaderFormatError(
            context=COUNTS_CONTEXT,
            path=path,
            found="no feature columns",
            expected="a cell-ID field and at least one count feature",
        )
    # Feature names become the table's var names; Arrow keeps empty and NUL names.
    if any(not name or "\x00" in name for name in features):
        raise ReaderFormatError(
            context=COUNTS_CONTEXT,
            path=path,
            found="empty or NUL-containing feature names",
            expected="nonempty feature names without NUL",
        )
    return pa.schema([(header[0], CELL_ID), *((name, COUNT) for name in features)])


def validate_cell_columns(header: Sequence[str], path: ArtifactFile) -> None:
    """Require the projected cell fields; any column order and extra annotations are accepted."""
    if not set(CELL_TYPES).issubset(header):
        raise ReaderFormatError(
            context=CELLS_CONTEXT,
            path=path,
            found=f"columns {list(header)!r}",
            expected=f"cell fields {tuple(CELL_TYPES)!r}",
        )


def transcript_types(header: Sequence[str], path: ArtifactFile) -> dict[str, pa.DataType]:
    """Return the projected transcript fields and their types; additional vendor annotations are ignored."""
    if not set(_TRANSCRIPT_REQUIRED).issubset(header):
        raise ReaderFormatError(
            context=TRANSCRIPTS_CONTEXT,
            path=path,
            found=f"columns {list(header)!r}",
            expected="transcript fields name, x and y",
        )
    return {name: kind for name, kind in TRANSCRIPT_TYPES.items() if name in _TRANSCRIPT_REQUIRED or name in header}


def normalize_transcript_partition(frame: pd.DataFrame, *, features: pd.Index, path: ArtifactFile) -> pd.DataFrame:
    """Validate one owned typed partition without scanning any other source rows.

    ``name`` becomes categorical with ``features`` as categories, and ``x`` and
    ``y`` become NumPy float64, as :class:`~spatialdata.models.PointsModel`
    requires; ``cell`` keeps its Arrow type and ``z`` becomes ``z_raw``.
    """
    codes = features.get_indexer(pd.Index(frame["name"]))
    if (codes < 0).any():
        raise ReaderFormatError(
            context=TRANSCRIPTS_CONTEXT,
            path=path,
            found="unknown transcript names",
            expected="names in the expression vocabulary",
        )
    frame["name"] = pd.Categorical.from_codes(codes, categories=features)
    for axis in ("x", "y"):
        values = frame[axis].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ReaderFormatError(
                context=TRANSCRIPTS_CONTEXT,
                path=path,
                found=f"nonfinite {axis!r} values",
                expected="finite coordinates",
            )
        frame[axis] = values
    if "z" in frame:
        # Vendor z is a nonnegative image-plane index, not physical distance.
        frame["z_raw"] = frame.pop("z")
    return frame


class BoundaryFeature(TypedDict):
    """Retain one source feature with authoritative top-level identity.

    ``geometry`` is validated and constructed by the shared GeoJSON polygon boundary.
    """

    type: Literal["Feature"]
    id: str | int
    geometry: object
    properties: dict[str, object] | None


def validate_cell_ids(ids: pa.Array | pa.ChunkedArray, *, context: ReaderErrorContext, path: ArtifactFile) -> pd.Index:
    """Return positive unique typed cell IDs as an Arrow-backed index."""
    index = pd.Index(pd.arrays.ArrowExtensionArray(ids))
    if len(index) and index.min() == 0:
        raise ReaderFormatError(context=context, path=path, found="nonpositive cell IDs", expected="positive cell IDs")
    if not index.is_unique:
        raise ReaderFormatError(context=context, path=path, found="duplicate cell IDs", expected="unique cell IDs")
    return index


def boundary_features(document: dict[str, object], path: ArtifactFile) -> tuple[list[BoundaryFeature], pd.Index]:
    """Validate the Feature schema and cast the independent top-level feature IDs to the cell-ID type."""
    features = document.get("features")
    if document.get("type") != "FeatureCollection" or not isinstance(features, list) or not features:
        raise ReaderFormatError(
            context=BOUNDARIES_CONTEXT,
            path=path,
            found="empty or malformed boundary collection",
            expected="a nonempty FeatureCollection",
        )
    raw_ids: list[str] = []
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
        raw_ids.append(str(identifier))
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
    expected = "unique positive uint64 feature ids"
    try:
        identifiers = pc.cast(pa.array(raw_ids, pa.string()), CELL_ID)
    except pa.ArrowInvalid as error:
        raise ReaderFormatError(
            context=BOUNDARIES_CONTEXT,
            path=path,
            found=f"feature ids that are not integers ({error})",
            expected=expected,
        ) from error
    index = validate_cell_ids(identifiers, context=BOUNDARIES_CONTEXT, path=path)
    # The schema above validates feature identity and properties. Geometry
    # structure belongs to the shared GeoJSON polygon constructor.
    return cast("list[BoundaryFeature]", features), index
