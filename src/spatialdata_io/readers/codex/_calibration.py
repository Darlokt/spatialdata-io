"""Explicit CODEX acquisition calibration and selected-image registration."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile


def validate_calibration(pixel_size_um: float | None, image_to_table: Affine | None) -> None:
    """Validate external scale and affine exactly once before loading cells."""
    if pixel_size_um is not None:
        if isinstance(pixel_size_um, bool) or not isinstance(pixel_size_um, (int, float)):
            msg = "pixel_size_um must be a positive finite number or None."
            raise TypeError(msg)
        if not math.isfinite(pixel_size_um) or pixel_size_um <= 0:
            msg = "pixel_size_um must be a positive finite number or None."
            raise ValueError(msg)
    if image_to_table is not None:
        if not isinstance(image_to_table, Affine):
            msg = "image_to_table must be an Affine or None."
            raise TypeError(msg)
        if image_to_table.input_axes != ("x", "y") or image_to_table.output_axes != ("x", "y"):
            msg = "image_to_table must map exactly XY input axes to XY output axes."
            raise ValueError(msg)
        # The general transform conversion reconstructs its bottom row; inspect
        # the caller's original matrix so near-projective terms cannot disappear.
        matrix = image_to_table.matrix
        if (
            matrix.shape != (3, 3)
            or matrix.dtype != np.dtype(np.float64)
            or not np.isfinite(matrix).all()
            or not np.array_equal(matrix[2], [0, 0, 1])
        ):
            msg = "image_to_table must be a finite float64 3-by-3 affine matrix."
            raise ValueError(msg)
        if np.linalg.slogdet(matrix[:2, :2])[0] == 0:
            msg = "image_to_table must be invertible."
            raise ValueError(msg)


def read_experiment_scale(path: ArtifactFile) -> float:
    """Read a caller-associated Akoya 1.5/1.7 acquisition declaration in nm/pixel."""
    context = ReaderErrorContext("codex", "experiment calibration")
    if path.stat().st_size > 1024 * 1024:
        raise ReaderFormatError(context=context, path=path, found="over 1 MiB", expected="bounded experiment JSON")
    metadata = read_json_object(path, context=context)
    version = metadata.get("version")
    value = metadata.get("xyResolution")
    if not isinstance(version, str) or version not in {"1.5.0.38", "1.7.0.6"}:
        raise ReaderFormatError(
            context=context, path=path, found=repr(version), expected="Akoya version 1.5.0.38 or 1.7.0.6"
        )
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=repr(value), expected="positive finite xyResolution in nm/pixel"
        )
    scale = value / 1000
    if scale == 0:
        raise ReaderFormatError(
            context=context, path=path, found=repr(value), expected="representable positive micrometre scale"
        )
    return scale
