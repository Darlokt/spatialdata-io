"""Invert the authoritative micron-to-mosaic-pixel calibration once."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING

import numpy as np
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

_MAX_TRANSFORM_BYTES = 64 * 1024
_AFFINE_SHAPE = (3, 3)


def read_transform(path: ArtifactFile) -> Affine:
    """Return the validated inverse mapping mosaic pixels into source micrometres."""
    context = ReaderErrorContext("merscope", "micron-to-mosaic transform")
    try:
        with path.open("rb") as source:
            payload = source.read(_MAX_TRANSFORM_BYTES + 1)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    if len(payload) > _MAX_TRANSFORM_BYTES:
        raise ReaderFormatError(
            context=context, path=path, found="oversized transform file", expected="a 3x3 space-delimited affine matrix"
        )
    try:
        matrix = _parse_matrix(payload)
    except (UnicodeError, ValueError, InvalidOperation, OverflowError) as error:
        raise ReaderFormatError(
            context=context,
            path=path,
            found="malformed transform values",
            expected="a 3x3 space-delimited numeric matrix",
        ) from error
    if not np.array_equal(matrix[-1], [0, 0, 1]):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"matrix shape/values {matrix!r}",
            expected="a finite 3x3 matrix with homogeneous row [0, 0, 1]",
        )
    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as error:
        raise ReaderFormatError(
            context=context, path=path, found="singular XY calibration", expected="an invertible affine transform"
        ) from error
    if not np.isfinite(inverse).all():
        raise ReaderFormatError(
            context=context,
            path=path,
            found="nonfinite inverse calibration",
            expected="a finite float64 inverse affine",
        )
    return Affine(inverse, input_axes=("x", "y"), output_axes=("x", "y"))


def _parse_matrix(payload: bytes) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
    """Parse exactly nine bounded textual coefficients in row-major order."""
    rows = [line.split() for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != _AFFINE_SHAPE[0] or any(len(row) != _AFFINE_SHAPE[1] for row in rows):
        message = "Expected exactly three rows of three coefficients."
        raise ValueError(message)
    values = [_coefficient(token) for row in rows for token in row]
    return np.asarray(values, dtype=np.float64).reshape(_AFFINE_SHAPE)


def _coefficient(token: str) -> float:
    """Decode one small coefficient without underflow or integral precision loss."""
    number = float(token)
    exact = Decimal(token)
    if (
        not np.isfinite(number)
        or (number == 0 and exact != 0)
        or (exact == exact.to_integral_value() and exact != Decimal(number))
    ):
        message = "Transform coefficient is nonfinite or loses its integral value or nonzero magnitude."
        raise ValueError(message)
    return number
