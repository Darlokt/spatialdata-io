"""Bounded scale-factor and HDF5 metadata parsing for Visium."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.visium._layout import H5Counts

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile
    from spatialdata_io.readers.visium._layout import CountsLayout

MAX_SCALE_FACTOR_BYTES = 64 * 1024
MAX_METADATA_TEXT_LENGTH = 4_096
HIRES_SCALE = "tissue_hires_scalef"
LOWRES_SCALE = "tissue_lowres_scalef"
SPOT_DIAMETER = "spot_diameter_fullres"


@dataclass(frozen=True, slots=True)
class ScaleFactors:
    """Validated transformations plus the complete owned vendor mapping."""

    hires: float
    lowres: float
    spot_diameter: float
    values: dict[str, object]


@dataclass(frozen=True, slots=True)
class CountMetadata:
    """Small reader-owned metadata copied from a closed HDF5 source."""

    library_id: str | None
    chemistry_description: str | None
    software_version: str | None


def _context(artifact: str) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium", artifact=artifact)


def _positive_number(values: dict[str, object], key: str, *, path: ArtifactFile) -> float:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReaderFormatError(
            context=_context("scale factors"),
            found=f"{key!r}={value!r}",
            expected=f"a finite positive numeric {key!r}",
            path=path,
        )
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ReaderFormatError(
            context=_context("scale factors"),
            found=f"{key!r}={value!r}",
            expected=f"a finite positive numeric {key!r}",
            path=path,
        )
    return result


def read_scale_factors(path: ArtifactFile) -> ScaleFactors:
    """Read and validate the bounded Space Ranger scale-factor object."""
    values = read_json_object(path, max_bytes=MAX_SCALE_FACTOR_BYTES, context=_context("scale factors"))
    return ScaleFactors(
        hires=_positive_number(values, HIRES_SCALE, path=path),
        lowres=_positive_number(values, LOWRES_SCALE, path=path),
        spot_diameter=_positive_number(values, SPOT_DIAMETER, path=path),
        values=values,
    )


def _text(value: object, *, attribute: str, path: ArtifactFile) -> str:
    if isinstance(value, bytes | np.bytes_):
        try:
            result = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ReaderFormatError(
                context=_context("count metadata"),
                found=f"non-UTF-8 bytes in attribute {attribute!r}",
                expected="bounded UTF-8 text",
                path=path,
            ) from error
    elif isinstance(value, str | np.str_):
        result = str(value)
    else:
        raise ReaderFormatError(
            context=_context("count metadata"),
            found=f"attribute {attribute!r} with value type {type(value).__name__}",
            expected="bounded UTF-8 text",
            path=path,
        )
    if not result or len(result) > MAX_METADATA_TEXT_LENGTH:
        raise ReaderFormatError(
            context=_context("count metadata"),
            found=f"attribute {attribute!r} with {len(result)} characters",
            expected=f"nonempty text of at most {MAX_METADATA_TEXT_LENGTH} characters",
            path=path,
        )
    return result


def _optional_text(attributes: h5py.AttributeManager, name: str, *, path: ArtifactFile) -> str | None:
    return None if name not in attributes else _text(attributes[name], attribute=name, path=path)


def _library_id(attributes: h5py.AttributeManager, *, path: ArtifactFile) -> str | None:
    if "library_ids" not in attributes:
        return None
    raw: object = attributes["library_ids"]
    values = raw.reshape(-1).tolist() if isinstance(raw, np.ndarray) else [raw]
    if len(values) != 1:
        raise ReaderFormatError(
            context=_context("count metadata"),
            found=f"{len(values)} library IDs",
            expected="exactly one library ID from single-library spaceranger count output",
            path=path,
        )
    return _text(values[0], attribute="library_ids", path=path)


def read_count_metadata(counts: CountsLayout) -> CountMetadata:
    """Copy bounded optional HDF5 metadata, or return empty MEX metadata."""
    if not isinstance(counts, H5Counts):
        return CountMetadata(library_id=None, chemistry_description=None, software_version=None)
    try:
        with h5py.File(counts.path, mode="r") as source:
            return CountMetadata(
                library_id=_library_id(source.attrs, path=counts.path),
                chemistry_description=_optional_text(source.attrs, "chemistry_description", path=counts.path),
                software_version=_optional_text(source.attrs, "software_version", path=counts.path),
            )
    except OSError as error:
        add_reader_context(error, context=_context("count metadata"), path=counts.path)
        raise


def resolve_dataset_id(explicit: str | None, counts: CountsLayout, metadata: CountMetadata) -> str:
    """Resolve and validate the stable element/coordinate-system identifier."""
    inferred = metadata.library_id or counts.filename_prefix
    value = explicit if explicit is not None else inferred
    if value is None:
        message = "dataset_id is required when the selected count source contains no library identifier."
        raise ValueError(message)
    if not isinstance(value, str):
        message = f"dataset_id must be a string or None; found {type(value).__name__}."
        raise TypeError(message)
    if not value or value != value.strip() or any(character in value for character in ("/", "\\", "\x00")):
        message = "dataset_id must be nonempty, trimmed, and contain no path separators or NUL characters."
        raise ValueError(message)
    return value
