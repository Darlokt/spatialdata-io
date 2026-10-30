"""Steinbock H5AD annotations, compound IDs, centroids, and metadata contracts.

Required observation fields are named and unordered; additional obs/var fields
retain their original storage and missingness. Centroids are finite numeric
pixel coordinates in regionprops YX order. Generated IDs use exact signed or
unsigned 64-bit integers; region IDs are categorical strings. Conflicting
reserved output metadata and ambiguous annotation names are rejected.
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

from spatialdata_io._positions import POSITIONS_KEY
from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import anndata as ad

    from spatialdata_io.readers._utils.paths import ArtifactFile

_OBJECT_NAME = re.compile(r"Object ([0-9]+) in (.+)")
REQUIRED_OBSERVATION_COLUMNS = frozenset({"image", "centroid-0", "centroid-1"})
SIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.int64())
UNSIGNED_CELL_ID_DTYPE = pd.ArrowDtype(pa.uint64())


@dataclass(frozen=True, slots=True)
class LabelGrid:
    """Carry independently inspected label extents and the exact integer dtype."""

    shape: tuple[int, int]
    dtype: np.dtype[np.generic]


def _format_error(path: ArtifactFile, *, found: str, expected: str) -> ReaderFormatError:
    """Describe a cell-table schema failure with the originating artifact."""
    return ReaderFormatError(
        context=ReaderErrorContext("steinbock", "cell table"), path=path, found=found, expected=expected
    )


def _validate_table(table: ad.AnnData, path: ArtifactFile) -> None:
    """Validate imported eager annotations before selecting named source fields."""
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
    required = REQUIRED_OBSERVATION_COLUMNS
    if not required.issubset(obs.columns):
        raise _format_error(path, found=f"columns {obs.columns.tolist()}", expected=f"columns {sorted(required)}")


def _instance_identifiers(obs: pd.DataFrame, samples: Mapping[str, LabelGrid], path: ArtifactFile) -> list[int]:
    """Parse exact compound identifiers and require every discovered sample in the table."""
    source_names = {f"{name}.tiff": name for name in samples}
    if obs["image"].isna().any() or set(obs["image"]) != set(source_names):
        raise _format_error(
            path, found="table/image/mask sample disagreement", expected="exactly the discovered sample set"
        )

    # The upstream compound name includes the complete filename, including spaces.
    # Parse decimal IDs directly: a float intermediate cannot retain uint64 values.
    maxima = {f"{name}.tiff": np.iinfo(sample.dtype.str).max for name, sample in samples.items()}
    identifiers: list[int] = []
    for observation, source_name in zip(obs.index, obs["image"], strict=True):
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


def _centroid_columns(obs: pd.DataFrame, path: ArtifactFile) -> tuple[np.ndarray, np.ndarray]:
    """Extract nonnull real XY source arrays without changing imported annotations."""
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


def _validate_centroids(
    coordinates: np.ndarray, obs: pd.DataFrame, samples: Mapping[str, LabelGrid], path: ArtifactFile
) -> None:
    """Require finite XY pixel centers within each cell's independently inspected frame."""
    if not np.isfinite(coordinates).all():
        raise _format_error(path, found="invalid centroids", expected="finite real centroid coordinates")

    # Group once rather than rescanning the entire cell population for each image.
    for filename, rows in obs.groupby("image", sort=False, observed=True).indices.items():
        positions = coordinates[rows]
        height, width = samples[str(filename).removesuffix(".tiff")].shape
        if (positions < 0).any() or (positions[:, 0] > width - 1).any() or (positions[:, 1] > height - 1).any():
            raise _format_error(
                path,
                found=f"centroids outside {filename!r}",
                expected=f"pixel centers within YX extent {(height, width)}",
            )


def _metadata_columns(obs: pd.DataFrame, var: pd.DataFrame, path: ArtifactFile) -> list[str]:
    """Validate optional redundant image metadata and return collision-free feature keys."""
    if "Image" in obs and not obs["Image"].eq(obs["image"]).fillna(value=False).all():
        raise _format_error(path, found="conflicting Image and image columns", expected="equal source-image metadata")

    if any(not isinstance(name, str) or not name for name in var.columns):
        raise _format_error(path, found="empty or nonstring feature metadata keys", expected="nonempty string keys")
    columns = [
        ("Final Concentration" if name == "Final Concentration / Dilution" else name).replace(" ", "_")
        for name in var.columns
    ]
    if len(set(columns)) != len(columns):
        raise _format_error(path, found="feature metadata cleanup collision", expected="unique normalized column names")

    return columns


def _validate_output_column(obs: pd.DataFrame, key: str, values: pd.Series, *, path: ArtifactFile) -> None:
    """Reject conflicting reserved metadata using exact scalar comparisons.

    Equal existing values are accepted; the generated field takes its declared
    output storage. Python scalar comparison avoids float promotion of uint64 IDs.
    """
    if key in obs and any(
        not pd.api.types.is_scalar(left)
        or pd.isna(left)
        or isinstance(left, bool) != isinstance(right, bool)
        or left != right
        for left, right in zip(obs[key].tolist(), values.tolist(), strict=True)
    ):
        raise _format_error(
            path,
            found=f"conflicting reserved observation column {key!r}",
            expected="absent or semantically equal reader output metadata",
        )


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Own one selected emitted image's dimensions and optional IMC XY spacing."""

    shape: tuple[int, int]
    spacing_um: tuple[float, float] | None


_IMAGE_FIELDS = frozenset({"image", "width_px", "height_px"})
_EXTENT_FIELDS = frozenset({"acquisition_width_um", "acquisition_height_um"})
_IMAGE_METADATA_CONTEXT = ReaderErrorContext("steinbock", "IMC images.csv")


def image_metadata_columns(header: Sequence[str], path: ArtifactFile) -> dict[str, int]:
    """Validate unique named fields and return projected source positions."""
    if (
        len(header) != len(set(header))
        or not _IMAGE_FIELDS.issubset(header)
        or len(_EXTENT_FIELDS.intersection(header)) == 1
    ):
        raise ReaderFormatError(
            context=_IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"header {list(header)!r}",
            expected=(
                "unique named image, width_px, height_px fields and either both acquisition extent fields or neither"
            ),
        )
    return {name: index for index, name in enumerate(header) if name in _IMAGE_FIELDS | _EXTENT_FIELDS}


def _metadata_number(text: str, path: ArtifactFile, *, image: str, field: str, integral: bool) -> Decimal:
    """Parse exact decimal text and bound its magnitude before any exact rational use.

    Decimal comparisons and float conversion reject huge exponents cheaply;
    building a Fraction from unchecked text such as ``1e999999999`` would not.
    """
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ReaderFormatError(
            context=_IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"{image!r} {field}={text!r}",
            expected="finite positive numeric metadata",
        ) from error
    if (
        not value.is_finite()
        or value <= 0
        or (integral and (value != value.to_integral_value() or value > 2**63 - 1))
        or (not integral and (not isfinite(float(value)) or float(value) <= 0))
    ):
        raise ReaderFormatError(
            context=_IMAGE_METADATA_CONTEXT,
            path=path,
            found=f"{image!r} {field}={text!r}",
            expected="positive integral dimensions or finite positive physical extents",
        )
    return value


def normalize_image_metadata(
    columns: Mapping[str, Sequence[str]], selected: set[str], path: ArtifactFile
) -> dict[str, ImageMetadata]:
    """Validate identities globally and selected rows' emitted dimensions and extent pairs."""
    names = columns["image"]
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ReaderFormatError(
            context=_IMAGE_METADATA_CONTEXT,
            path=path,
            found="empty or duplicate image names",
            expected="unique exact image basenames",
        )
    result: dict[str, ImageMetadata] = {}
    for index, name in enumerate(names):
        if name not in selected:
            continue
        width = int(_metadata_number(columns["width_px"][index], path, image=name, field="width_px", integral=True))
        height = int(_metadata_number(columns["height_px"][index], path, image=name, field="height_px", integral=True))
        spacing = None
        if "acquisition_width_um" in columns:
            x = columns["acquisition_width_um"][index]
            y = columns["acquisition_height_um"][index]
            if bool(x) != bool(y):
                raise ReaderFormatError(
                    context=_IMAGE_METADATA_CONTEXT,
                    path=path,
                    found=f"partial physical extents for {name!r}",
                    expected="both acquisition extents or two empty fields",
                )
            if x:
                # Exact rational division avoids ambient Decimal-context rounding.
                width_um = _metadata_number(x, path, image=name, field="acquisition_width_um", integral=False)
                height_um = _metadata_number(y, path, image=name, field="acquisition_height_um", integral=False)
                spacing = (float(Fraction(width_um) / width), float(Fraction(height_um) / height))
                if not all(isfinite(value) and value > 0 for value in spacing):
                    raise ReaderFormatError(
                        context=_IMAGE_METADATA_CONTEXT,
                        path=path,
                        found=f"unrepresentable spacing for {name!r}",
                        expected="finite positive micrometres per pixel",
                    )
        result[name] = ImageMetadata((height, width), spacing)
    return result


def _validate_position_destinations(table: ad.AnnData, *, multiple: bool, path: ArtifactFile) -> None:
    """Reject occupied destinations before changing imported annotations."""
    if (
        POSITIONS_KEY in table.uns
        or "spatial_pixels" in table.obsm
        or ("spatial" in table.obsm and "spatial_source" in table.obsm)
        or (multiple and any(name in table.obs for name in ("spatial_frame", "spatial_pixel_frame")))
    ):
        raise ReaderFormatError(
            context=ReaderErrorContext("steinbock", "positions"),
            path=path,
            found="existing position declaration, preservation destination or generated frame column",
            expected="unoccupied position destinations; numerical equality does not establish source semantics",
        )
