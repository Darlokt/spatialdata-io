"""Schemas of the Steinbock AnnData export (``table_file``, by default ``cells.h5ad``) and of ``images.csv``.

``steinbock export anndata`` builds one AnnData per image and concatenates them
with ``label="Image"`` and ``index_unique=" in "``. The fields the reader uses::

    X                   objects x channels (mean intensities), dense or sparse
    obs_names           "Object <id> in <image>.tiff"; <id> is the mask label
    obs["image"]        "<image>.tiff", the object's source image       (required)
    obs["Image"]        concatenation label, equal to obs["image"]      (optional)
    obs["centroid-0"]   regionprops centroid row (y), zero-based pixels (required)
    obs["centroid-1"]   regionprops centroid column (x)                 (required)
    var_names           channel names; var columns come from panel.csv

Named fields may come in any order; every other ``obs``/``var`` column, layer,
mapping and ``raw`` is kept with its dtype and missingness. Column names and
observation/feature IDs must be unique, and feature IDs nonempty. ``<id>`` is a
positive decimal integer within the mask dtype (zero is background); IDs repeat
across images but not within one. Centroids follow scikit-image's ``(row,
col)`` order and may be any integer or floating dtype, including nullable and
Arrow ones without missing values; integer and float centroids must combine
without rounding. The destinations the reader generates (``cell_id``,
``region``, ``obsm["spatial_pixels"]``, the position declaration and, for
several samples, the frame columns) must not already exist.

``images.csv`` is written by ``steinbock preprocess imc images`` (or by hand
for other sources), one row per image::

    image, width_px, height_px, ..., acquisition_width_um, acquisition_height_um, ...
    sample - a.tiff, 600, 500, ..., 600.0, 500.0, ...

``image``, ``width_px`` and ``height_px`` are required, the two acquisition
extents (micrometres, from the IMC ROI) are optional but come as a pair; other
columns are ignored. Arrow reads the supported fields as UTF-8 strings (a
byte-order mark is allowed). Every record has the header's width; NUL is
rejected in all header names and supported fields, while ignored extra values
are not scanned. Image names are unique and nonempty in every row; numbers are
interpreted only in discovered samples. Dimensions use checked uint64 parsing
with a positive int64 bound; extents retain exact Decimal/Fraction arithmetic
and are either both empty (no IMC calibration) or both finite and positive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.steinbock._images import SampleGrid

CELL_TABLE_CONTEXT = ReaderErrorContext("steinbock", "cell table")
IMAGE_METADATA_CONTEXT = ReaderErrorContext("steinbock", "images.csv")
POSITIONS_CONTEXT = ReaderErrorContext("steinbock", "positions")
REQUIRED_OBSERVATION_COLUMNS = frozenset({"image", "centroid-0", "centroid-1"})
SIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.int64())
UNSIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.uint64())

_OBJECT_NAME = re.compile(r"Object ([0-9]+) in (.+)")
_IMAGE_FIELDS = frozenset({"image", "width_px", "height_px"})
_EXTENT_FIELDS = frozenset({"acquisition_width_um", "acquisition_height_um"})
_LEGACY_DILUTION = "Final_Concentration"


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """One discovered image's ``images.csv`` dimensions ``(height, width)`` and optional XY spacing."""

    shape: tuple[int, int]
    spacing_um: tuple[float, float] | None


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a cell-table schema failure."""
    return ReaderFormatError(context=CELL_TABLE_CONTEXT, path=path, found=found, expected=expected)


def validate_table(table: ad.AnnData, path: ArtifactFile) -> None:
    """Validate the imported identifiers and named fields before anything is changed."""
    if table.X is None:
        raise _format_error(path, found="missing expression X", expected="an observation-by-feature expression array")
    if not isinstance(table.obs, pd.DataFrame) or not isinstance(table.var, pd.DataFrame):
        message = "Steinbock requires eager DataFrames from read_h5ad."
        raise TypeError(message)

    if not table.obs.columns.is_unique or not table.var.columns.is_unique:
        raise _format_error(
            path, found="duplicate annotation columns", expected="unique observation and feature column names"
        )
    if (
        not table.obs_names.is_unique
        or not table.var_names.is_unique
        or table.obs_names.hasnans
        or table.var_names.hasnans
        or bool(table.var_names.isin([""]).any())
    ):
        raise _format_error(path, found="duplicate or empty identifiers", expected="unique observation and feature IDs")

    obs = table.obs
    if not REQUIRED_OBSERVATION_COLUMNS.issubset(obs.columns):
        raise _format_error(
            path, found=f"columns {obs.columns.tolist()}", expected=f"columns {sorted(REQUIRED_OBSERVATION_COLUMNS)}"
        )
    occupied = [name for name in ("cell_id", "region") if name in obs]
    if occupied:
        raise _format_error(
            path,
            found=f"occupied generated observation columns {occupied!r}",
            expected="unoccupied cell_id and region destinations",
        )


def validate_position_destinations(table: ad.AnnData, *, multiple: bool, path: ArtifactFile) -> None:
    """Reject occupied position destinations before changing imported annotations."""
    if (
        POSITIONS_KEY in table.uns
        or "spatial_pixels" in table.obsm
        or ("spatial" in table.obsm and "spatial_source" in table.obsm)
        or (multiple and any(name in table.obs for name in ("spatial_frame", "spatial_pixel_frame")))
    ):
        raise ReaderFormatError(
            context=POSITIONS_CONTEXT,
            path=path,
            found="existing position declaration, preservation destination or generated frame column",
            expected="unoccupied position destinations; numerical equality does not establish source semantics",
        )


def instance_identifiers(obs: pd.DataFrame, grids: Mapping[str, SampleGrid], path: ArtifactFile) -> list[int]:
    """Parse the exact mask label of every observation name and require exactly the discovered samples."""
    maxima = {grid.image_name: np.iinfo(grid.dtype).max for grid in grids.values()}
    if obs["image"].isna().any() or set(obs["image"]) != set(maxima):
        raise _format_error(
            path, found="table/image/mask sample disagreement", expected="exactly the discovered sample set"
        )

    # The compound name holds the complete file name, which may contain " in ".
    # Python integers keep uint64 labels exact, which a float detour would not.
    identifiers: list[int] = []
    for observation, source_name in zip(obs.index.tolist(), obs["image"].tolist(), strict=True):
        match = _OBJECT_NAME.fullmatch(observation)
        if match is None or match[2] != source_name:
            raise _format_error(
                path, found=f"observation {observation!r}", expected="Object <positive integer> in <obs.image>"
            )
        identifier = int(match[1])
        if identifier <= 0 or identifier > maxima[source_name]:
            raise _format_error(
                path,
                found=f"label ID {identifier} for {source_name!r}",
                expected="a positive ID fitting its mask dtype",
            )
        identifiers.append(identifier)
    return identifiers


def centroid_columns(obs: pd.DataFrame, path: ArtifactFile) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y centroid arrays in their native numeric dtypes."""
    columns: list[np.ndarray] = []
    for name in ("centroid-1", "centroid-0"):
        column = obs[name]
        if (
            column.isna().any()
            or pd.api.types.is_bool_dtype(column.dtype)
            or not (pd.api.types.is_integer_dtype(column.dtype) or pd.api.types.is_float_dtype(column.dtype))
        ):
            raise _format_error(path, found="invalid centroids", expected="nonnull real numeric centroid columns")

        # Nullable pandas/Arrow numerics otherwise produce object arrays. Their
        # native dtype preserves width and signedness without a float detour.
        dtype = (
            np.dtype(column.dtype.pyarrow_dtype.to_pandas_dtype())
            if isinstance(column.dtype, pd.ArrowDtype)
            else np.dtype(str(column.dtype).lower())
        )
        columns.append(column.to_numpy(dtype=dtype, copy=False))

    # Combining integer columns with other numeric dtypes can choose floating
    # storage. Check only integers beyond its exact range before conversion.
    common = np.result_type(columns[0].dtype, columns[1].dtype)
    if common.kind == "f":
        precision = np.finfo(common).nmant + 1
        for values in columns:
            if values.dtype.kind in "iu" and np.iinfo(values.dtype).bits > precision:
                limit = 2**precision
                candidates = values > limit
                if values.dtype.kind == "i":
                    candidates |= values < -limit
                if any(int(common.type(value)) != int(value) for value in values[candidates]):
                    raise _format_error(
                        path, found="centroid integer precision loss", expected="lossless common numeric XY storage"
                    )
    return columns[0], columns[1]


def validate_centroids(
    coordinates: np.ndarray, obs: pd.DataFrame, grids: Mapping[str, SampleGrid], path: ArtifactFile
) -> None:
    """Require finite XY pixel centres within each object's own mask extent."""
    if not np.isfinite(coordinates).all():
        raise _format_error(path, found="invalid centroids", expected="finite real centroid coordinates")

    shapes = {grid.image_name: grid.shape for grid in grids.values()}
    # Group once rather than rescanning the entire cell population for each image.
    for filename, rows in obs.groupby("image", sort=False, observed=True).indices.items():
        positions = coordinates[rows]
        height, width = shapes[str(filename)]
        if (positions < 0).any() or (positions[:, 0] > width - 1).any() or (positions[:, 1] > height - 1).any():
            raise _format_error(
                path,
                found=f"centroids outside {filename!r}",
                expected=f"pixel centers within YX extent {(height, width)}",
            )


def metadata_columns(obs: pd.DataFrame, var: pd.DataFrame, path: ArtifactFile) -> list[str]:
    """Check the optional ``Image`` column and return the feature column names with the legacy dilution name.

    ``Final Concentration / Dilution`` becomes ``Final_Concentration``; other
    names are returned unchanged for the shared column-name sanitizer.
    """
    if "Image" in obs and not obs["Image"].eq(obs["image"]).fillna(value=False).all():
        raise _format_error(path, found="conflicting Image and image columns", expected="equal source-image metadata")

    if any(not isinstance(name, str) or not name for name in var.columns):
        raise _format_error(path, found="empty or nonstring feature metadata keys", expected="nonempty string keys")
    columns = [_LEGACY_DILUTION if name == "Final Concentration / Dilution" else name for name in var.columns]
    if len(set(columns)) != len(columns):
        raise _format_error(
            path,
            found=f"both 'Final Concentration / Dilution' and {_LEGACY_DILUTION!r}",
            expected="one dilution column",
        )
    return columns


def image_metadata_types(header: Sequence[str], path: ArtifactFile) -> pa.Schema:
    """Declare supported string fields in source order after unique header parsing."""
    if (
        any("\x00" in name for name in header)
        or not _IMAGE_FIELDS.issubset(header)
        or len(_EXTENT_FIELDS.intersection(header)) == 1
    ):
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"header {list(header)!r}",
            expected=(
                "NUL-free names, image, width_px, height_px and either both acquisition extent fields or neither"
            ),
        )
    return pa.schema([(name, pa.string()) for name in header if name in _IMAGE_FIELDS or name in _EXTENT_FIELDS])


def _physical_extent(text: str, path: ArtifactFile, *, image: str, field: str) -> Decimal:
    """Parse exact decimal text and bound its magnitude before any exact rational use.

    Decimal comparisons and float conversion reject huge exponents cheaply;
    building a Fraction from unchecked text such as ``1e999999999`` would not.
    """
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"{image!r} {field}={text!r}",
            expected="finite positive numeric metadata",
        ) from error
    if not value.is_finite() or value <= 0 or not isfinite(float(value)) or float(value) <= 0:
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"{image!r} {field}={text!r}",
            expected="finite positive physical extents",
        )
    return value


def normalize_image_metadata(table: pa.Table, selected: set[str], path: ArtifactFile) -> dict[str, ImageMetadata]:
    """Validate projected strings globally and interpret only selected image rows."""
    frame = table.to_pandas(types_mapper=pd.ArrowDtype)
    names = frame["image"]
    if names.eq("").any() or not names.is_unique:
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found="empty or duplicate image names",
            expected="unique exact image basenames",
        )
    if any(frame[name].str.contains("\x00", regex=False).any() for name in frame):
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found="NUL-containing metadata",
            expected="NUL-free supported fields",
        )
    frame = frame.set_index("image").loc[names.isin(selected).array]
    for field in ("width_px", "height_px"):
        try:
            # Arrow casts do not trim the spaces/tabs accepted by its CSV parser.
            values = pa.array(frame[field].str.strip(" \t"), type=pa.string())
            dimensions = pc.cast(values, pa.uint64(), safe=True)
        except pa.ArrowInvalid as error:
            raise ReaderFormatError(
                context=IMAGE_METADATA_CONTEXT,
                path=path,
                found=f"invalid {field} in selected image metadata",
                expected="positive integral dimensions in Arrow's uint64 grammar",
            ) from error
        bounds = pc.call_function("min_max", [dimensions]).as_py()
        if len(dimensions) and (bounds["min"] == 0 or bounds["max"] > 2**63 - 1):
            raise ReaderFormatError(
                context=IMAGE_METADATA_CONTEXT,
                path=path,
                found=f"{field} outside its supported range",
                expected="positive integral dimensions at most 9223372036854775807",
            )
        frame[field] = pd.arrays.ArrowExtensionArray(dimensions)

    fields = ["width_px", "height_px"]
    if "acquisition_width_um" in frame:
        fields.extend(("acquisition_width_um", "acquisition_height_um"))
    result: dict[str, ImageMetadata] = {}
    for name, width, height, *extents in frame[fields].itertuples(name=None):
        shape = (int(height), int(width))
        spacing = None
        if extents:
            x, y = extents
            spacing = _physical_spacing(x, y, shape, image=name, path=path)
        result[name] = ImageMetadata(shape, spacing)
    return result


def _physical_spacing(
    x: str, y: str, shape: tuple[int, int], *, image: str, path: ArtifactFile
) -> tuple[float, float] | None:
    """Divide exact decimal extents by integer dimensions before rounding the spacing."""
    if bool(x) != bool(y):
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"partial physical extents for {image!r}",
            expected="both acquisition extents or two empty fields",
        )
    if not x:
        return None
    # Exact rational division avoids ambient Decimal-context rounding.
    width_um = _physical_extent(x, path, image=image, field="acquisition_width_um")
    height_um = _physical_extent(y, path, image=image, field="acquisition_height_um")
    height, width = shape
    spacing = (float(Fraction(width_um) / width), float(Fraction(height_um) / height))
    if not all(isfinite(value) and value > 0 for value in spacing):
        raise ReaderFormatError(
            context=IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"unrepresentable spacing for {image!r}",
            expected="finite positive micrometres per pixel",
        )
    return spacing
