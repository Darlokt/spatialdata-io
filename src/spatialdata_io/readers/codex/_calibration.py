"""Resolve the CODEX table pixel size and validate the explicit image registration.

A cell table states positions in native region pixels and carries no pixel
size. The caller supplies one of two calibrations; nothing is inferred from
directory proximity, file names or image dimensions:

- ``pixel_size_um``: micrometres per native acquisition/table pixel;
- an explicitly associated Akoya ``experiment.json`` declaring ``xyResolution``
  in nm/pixel. Supported declarations, with their ``version`` field::

      {"version": "1.5.0.38", "xyResolution": 377.4463383838384, ...}
      {"version": "1.7.0.6",  "xyResolution": 377.40384615384613, ...}

  Examples: https://github.com/hubmapconsortium/codex-metadata-converters/tree/737f4fa15a5dfd126be9d0748647bee26c407dee/converter/examples

If both are given they must agree exactly. A stitched image has its own pixel
grid; only the caller's ``image_to_table`` affine relates it to table pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from spatialdata.transformations import Affine, Identity

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

type AkoyaVersion = Literal["1.5.0.38", "1.7.0.6"]

_CALLER_SOURCE = "Caller-declared micrometres per native acquisition/table pixel."
_EXPERIMENT_SOURCE = "Caller-associated Akoya experiment xyResolution in nm/pixel, divided by 1000."
_XY_AXES = ("x", "y")


@dataclass(frozen=True, slots=True)
class Calibration:
    """Positive finite micrometres per table pixel and where the value came from.

    Attributes
    ----------
    pixel_size_um
        Micrometres per native table pixel.
    source
        Human-readable origin, recorded in the positions metadata.
    version
        Akoya version of the experiment declaration, or ``None`` for a
        caller-declared pixel size.
    """

    pixel_size_um: float
    source: str
    version: AkoyaVersion | None


def _has_finite_inverse(matrix: NDArray[np.floating]) -> bool:
    """Check the inverse used to map XY back into this image, including its translation."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return False
    return bool(np.isfinite(inverse).all())


def validate_calibration(pixel_size_um: float | None, image_to_table: Affine | None) -> None:
    """Validate the public scale and affine before any I/O."""
    if pixel_size_um is not None:
        if isinstance(pixel_size_um, bool) or not isinstance(pixel_size_um, (int, float)):
            msg = "pixel_size_um must be a positive finite number or None."
            raise TypeError(msg)
        try:
            valid = pixel_size_um > 0 and math.isfinite(pixel_size_um)
        except OverflowError:
            # Python integers are unbounded, but the transformation uses float64.
            valid = False
        if not valid:
            msg = "pixel_size_um must be a positive finite number representable as float64 or None."
            raise ValueError(msg)
    if image_to_table is not None:
        if not isinstance(image_to_table, Affine):
            msg = "image_to_table must be an Affine or None."
            raise TypeError(msg)
        if image_to_table.input_axes != _XY_AXES or image_to_table.output_axes != _XY_AXES:
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
        if not _has_finite_inverse(matrix):
            msg = "image_to_table must be invertible."
            raise ValueError(msg)


def read_experiment(path: ArtifactFile) -> Calibration:
    """Read the pixel size declared by a supported Akoya experiment JSON."""
    context = ReaderErrorContext("codex", "experiment calibration")
    metadata = read_json_object(path, context=context)

    version: AkoyaVersion
    match metadata.get("version"):
        case "1.5.0.38" | "1.7.0.6" as declared:
            version = declared
        case other:
            raise ReaderFormatError(
                context=context, path=path, found=repr(other), expected="Akoya version 1.5.0.38 or 1.7.0.6"
            )
    context = ReaderErrorContext("codex", "experiment calibration", detected_version=version)

    # Strict JSON numbers are finite; booleans are ints in Python but never a size.
    value = metadata.get("xyResolution")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=repr(value), expected="positive xyResolution in nm/pixel"
        )
    size_um = value / 1000
    if size_um == 0:
        raise ReaderFormatError(
            context=context, path=path, found=repr(value), expected="representable positive micrometre scale"
        )
    return Calibration(size_um, _EXPERIMENT_SOURCE, version)


def resolve_calibration(pixel_size_um: float | None, experiment: ArtifactFile | None) -> Calibration | None:
    """Return the table calibration, requiring an exact match when both sources are given."""
    if experiment is None:
        return None if pixel_size_um is None else Calibration(pixel_size_um, _CALLER_SOURCE, None)

    declared = read_experiment(experiment)
    if pixel_size_um is not None and pixel_size_um != declared.pixel_size_um:
        msg = (
            f"pixel_size_um {pixel_size_um!r} conflicts with the experiment xyResolution / 1000 "
            f"({declared.pixel_size_um!r})."
        )
        raise ValueError(msg)
    return declared


def image_transformation(image_to_table: Affine | None, calibration: Calibration | None) -> Affine | Identity:
    """Return the image transform in its selected frame, with checked float64 composition."""
    if image_to_table is None:
        return Identity()
    if calibration is None:
        return image_to_table

    matrix = image_to_table.matrix.copy()
    with np.errstate(over="ignore", under="ignore"):
        matrix[:2] *= calibration.pixel_size_um
    if not np.isfinite(matrix).all() or not _has_finite_inverse(matrix):
        message = "image_to_table and the pixel size must produce a finite and invertible physical image transform."
        raise ValueError(message)
    return Affine(matrix, input_axes=_XY_AXES, output_axes=_XY_AXES)
