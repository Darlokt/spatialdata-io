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
from spatialdata.transformations import Affine

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.metadata import read_json_object

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

type AkoyaVersion = Literal["1.5.0.38", "1.7.0.6"]

_CALLER_SOURCE = "Caller-declared micrometres per native acquisition/table pixel."
_EXPERIMENT_SOURCE = "Caller-associated Akoya experiment xyResolution in nm/pixel, divided by 1000."


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


def validate_calibration(pixel_size_um: float | None, image_to_table: Affine | None) -> None:
    """Validate the public scale and affine before any I/O."""
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
    return Calibration(value / 1000, _EXPERIMENT_SOURCE, version)


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
