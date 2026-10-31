"""Space Ranger identity, calibration and feature-slice metadata."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Never, cast

import h5py
import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.metadata import read_json_object
from spatialdata_io.readers.visium_hd._transforms import SPOT_PITCH_UM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile


@dataclass(frozen=True, slots=True)
class Identity:
    """Optional source library identity and normalized Space Ranger version."""

    format_version: str
    library_id: str | None


@dataclass(frozen=True, slots=True)
class ScaleFactors:
    """Validated raster and bin calibration."""

    hires: float
    lowres: float
    microns_per_pixel: float


@dataclass(frozen=True, slots=True)
class SourceMetadata:
    """Small owned metadata copied from a closed feature-slice file."""

    nrows: int
    ncols: int
    cytassist: NDArray[np.float64] | None


def _context(artifact: str, version: str | None = None) -> ReaderErrorContext:
    return ReaderErrorContext(reader="visium_hd", artifact=artifact, detected_version=version)


def _error(
    path: ArtifactFile, found: str, expected: str, *, context: ReaderErrorContext | None = None
) -> ReaderFormatError:
    return ReaderFormatError(context=context or _context("metadata"), found=found, expected=expected, path=path)


def _text(value: object, *, path: ArtifactFile, context: ReaderErrorContext | None = None) -> str:
    if isinstance(value, bytes | np.bytes_):
        try:
            value = bytes(value).decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise _error(path, "non-UTF-8 metadata", "UTF-8 text", context=context) from error

    if not isinstance(value, str) or not value:
        raise _error(path, "missing or empty metadata text", "nonempty UTF-8 text", context=context)

    return value


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            msg = f"duplicate key {key!r}"
            raise ValueError(msg)
        result[key] = value

    return result


def _reject_constant(value: str) -> Never:
    msg = f"nonfinite JSON constant {value}"
    raise ValueError(msg)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        msg = "nonfinite JSON number"
        raise ValueError(msg)

    return parsed


def _json_object(value: object, *, path: ArtifactFile) -> dict[str, object]:
    try:
        parsed: object = json.loads(
            _text(value, path=path),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (ValueError, RecursionError) as error:
        raise _error(path, "malformed metadata JSON", "a JSON object with unique keys") from error

    if not isinstance(parsed, dict):
        raise _error(path, "non-object metadata JSON", "a JSON object")

    return cast("dict[str, object]", parsed)


def _positive(values: dict[str, object], name: str, *, path: ArtifactFile) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise _error(path, f"{name}={value!r}", "a finite positive number")
    return float(value)


def read_bin_scale_factors(path: ArtifactFile, size_um: int) -> ScaleFactors:
    """Read raster scales and validate the expected bin's nominal calibration."""
    values = read_json_object(path, context=_context("scale factors"))
    factors = ScaleFactors(
        hires=_positive(values, "tissue_hires_scalef", path=path),
        lowres=_positive(values, "tissue_lowres_scalef", path=path),
        microns_per_pixel=_positive(values, "microns_per_pixel", path=path),
    )
    diameter = _positive(values, "spot_diameter_fullres", path=path)
    bin_size = _positive(values, "bin_size_um", path=path)
    if bin_size != size_um:
        raise _error(path, f"bin_size_um={bin_size!r}", f"bin_size_um={size_um}")
    if not math.isclose(size_um / diameter, factors.microns_per_pixel, rel_tol=1e-12):
        raise _error(
            path,
            "inconsistent bin pitch, diameter and microns_per_pixel",
            "bin_size_um / spot_diameter_fullres = microns_per_pixel",
        )

    return factors


def read_identity(path: ArtifactFile) -> Identity:
    """Read a source's Space Ranger 3.0-4.1 version and optional library ID."""
    context = _context("count identity")
    try:
        with h5py.File(path, "r") as source:
            text = _text(source.attrs.get("software_version"), path=path, context=context)
            version = text.removeprefix("spaceranger-")
            context = _context("count identity", version)
            match = re.fullmatch(r"(?:spaceranger-)?((3|4)\.(\d+)\.(\d+))", text)
            if match is None or int(match[3]) > 1:
                raise _error(path, f"software_version={text!r}", "Space Ranger 3.0-4.1", context=context)

            raw: object = source.attrs.get("library_ids")
            if isinstance(raw, np.ndarray):
                if raw.size != 1:
                    raise _error(path, f"{raw.size} library IDs", "exactly one library ID", context=context)
                raw = raw.reshape(-1)[0]
            library_id = None if raw is None else _text(raw, path=path, context=context)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    return Identity(version, library_id)


def reconcile_identities(paths: Sequence[ArtifactFile]) -> Identity:
    """Read each distinct consumed source once and reconcile its export identity."""
    identity: Identity | None = None
    for path in dict.fromkeys(paths):
        candidate = read_identity(path)
        if identity is None:
            identity = candidate
            continue

        context = _context("count identity", candidate.format_version)
        if candidate.format_version != identity.format_version:
            raise _error(
                path,
                f"software_version={candidate.format_version!r}",
                f"software_version={identity.format_version!r} shared by all consumed sources",
                context=context,
            )
        if (
            identity.library_id is not None
            and candidate.library_id is not None
            and candidate.library_id != identity.library_id
        ):
            raise _error(
                path,
                f"library_id={candidate.library_id!r}",
                f"library_id={identity.library_id!r} shared by all known library IDs",
                context=context,
            )
        if identity.library_id is None and candidate.library_id is not None:
            identity = candidate

    if identity is None:
        raise ReaderFormatError(
            context=_context("count identity"),
            found="no selected identity source",
            expected="at least one selected identity-bearing artifact",
        )
    return identity


def _matrix(value: object, *, path: ArtifactFile, name: str) -> NDArray[np.float64] | None:
    if value is None:
        return None

    try:
        numeric = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as error:
        raise _error(path, f"malformed {name}", "a finite invertible 3 by 3 matrix or null") from error
    if numeric.shape not in {(9,), (3, 3)} or any(
        isinstance(element, bool) or not isinstance(element, int | float) for element in numeric.flat
    ):
        raise _error(path, f"malformed {name}", "a numeric 3 by 3 matrix or null")

    matrix = numeric.astype(np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise _error(path, f"nonfinite or singular {name}", "a finite invertible matrix")

    matrix.flags.writeable = False
    return matrix


def _dimension(values: dict[str, object], name: str, *, path: ArtifactFile) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(path, f"{name}={value!r}", "a positive integer grid dimension")

    return value


def _matrix_pair(values: dict[str, object], frame: str, *, path: ArtifactFile) -> NDArray[np.float64] | None:
    forward = f"spot_colrow_to_{frame}_colrow"
    inverse = f"{frame}_colrow_to_spot_colrow"
    if forward not in values or inverse not in values:
        raise _error(path, f"incomplete {frame} matrix pair", "both forward and inverse keys")

    first = _matrix(values[forward], path=path, name=forward)
    second = _matrix(values[inverse], path=path, name=inverse)
    if (first is None) != (second is None):
        raise _error(path, f"partly null {frame} matrix pair", "two matrices or two null values")
    if first is not None and second is not None:
        product = first @ second
        if product[2, 2] == 0 or not np.allclose(product / product[2, 2], np.eye(3), rtol=1e-6, atol=1e-6):
            raise _error(path, f"inconsistent {frame} matrix pair", "mutual inverses")
    return first


def read_source_metadata(path: ArtifactFile) -> SourceMetadata:
    """Validate feature metadata, retaining dimensions and the forward CytAssist map."""
    try:
        with h5py.File(path, "r") as source:
            values = _json_object(source.attrs.get("metadata_json"), path=path)
    except OSError as error:
        add_reader_context(error, context=_context("feature slice metadata"), path=path)
        raise

    matrices = values.get("transform_matrices")
    if not isinstance(matrices, dict):
        raise _error(path, "missing transform_matrices object", "nullable forward/inverse matrix pairs")

    _matrix_pair(matrices, "microscope", path=path)
    cytassist = _matrix_pair(matrices, "cytassist", path=path)

    pitch = _positive(values, "spot_pitch", path=path)
    if pitch != SPOT_PITCH_UM:
        raise _error(path, f"spot_pitch={pitch!r}", "the supported 2 micrometre HD grid pitch")
    _text(values.get("sample_id"), path=path)

    return SourceMetadata(
        _dimension(values, "nrows", path=path),
        _dimension(values, "ncols", path=path),
        cytassist,
    )
