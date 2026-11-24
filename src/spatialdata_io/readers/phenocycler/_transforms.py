"""Place QPI pixel grids and validate alternate bit depths of the same scan.

TIFF stage coordinates locate the upper-left pixel edge in micrometres. For
example, the centre of column ``j`` lies at ``stage_x + (j + 0.5) * pixel_size_x``.
With calibration alone the transform supplies a scale about the local origin.
A single calibrated scan uses ``global``. Several scans share that frame only
when all have calibration, stage positions and the same known slide identifier.
Otherwise each region keeps its own physical or pixel frame. A single
uncalibrated scan uses ``image_pixels``.

An ordinary 8-bit companion inherits the extended-range scan's transform
after their shapes, ordered channel identities, acquisition generation,
slide identifier, calibration and stage position have been checked. Explicit
mask-to-image and CODEX image-to-table registrations require invertible XY
affines with a finite float64 inverse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from spatialdata.transformations import Affine, Identity, Scale, Sequence, Translation

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers.phenocycler._images import matching_pixel_sizes, matching_stage_positions

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray
    from spatialdata.transformations import BaseTransformation

    from spatialdata_io.readers._utils.errors import ReaderErrorContext
    from spatialdata_io.readers.phenocycler._images import QpiScan

_XY = ("x", "y")


def has_finite_inverse(matrix: NDArray[np.floating]) -> bool:
    """Check the inverse used to map XY back into this image, including its translation."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return False
    return bool(np.isfinite(inverse).all())


def validate_xy_affine(affine: Affine, *, option: str) -> None:
    """Require a finite, invertible XY registration with an exact affine bottom row."""
    if not isinstance(affine, Affine):
        msg = f"{option} must be an Affine."
        raise TypeError(msg)
    if affine.input_axes != _XY or affine.output_axes != _XY:
        msg = f"{option} must map exactly XY input axes to XY output axes."
        raise ValueError(msg)
    # The general transform conversion reconstructs its bottom row; inspect
    # the caller's original matrix so near-projective terms cannot disappear.
    matrix = affine.matrix
    if (
        matrix.shape != (3, 3)
        or matrix.dtype != np.dtype(np.float64)
        or not np.isfinite(matrix).all()
        or not np.array_equal(matrix[2], [0, 0, 1])
    ):
        msg = f"{option} must be a finite float64 3-by-3 affine matrix."
        raise ValueError(msg)
    if not has_finite_inverse(matrix):
        msg = f"{option} must be invertible."
        raise ValueError(msg)


def scan_transformations(scans: Mapping[str, QpiScan]) -> dict[str, dict[str, BaseTransformation]]:
    """Resolve local or common-slide physical coordinates for selected primary scans."""
    slides = {scan.slide_id for scan in scans.values()}
    together = (
        len(slides) == 1
        and None not in slides
        and all(scan.pixel_size_um is not None and scan.stage_position_um is not None for scan in scans.values())
    )
    result: dict[str, dict[str, BaseTransformation]] = {}
    for region, scan in scans.items():
        if scan.pixel_size_um is None:
            result[region] = {f"{region}_pixels" if len(scans) > 1 else "image_pixels": Identity()}
        else:
            scale = Scale(list(scan.pixel_size_um), axes=_XY)
            transform: BaseTransformation = (
                scale
                if scan.stage_position_um is None
                else Sequence([scale, Translation(list(scan.stage_position_um), axes=_XY)])
            )
            frame = "global" if len(scans) == 1 or together else f"{region}_um"
            result[region] = {frame: transform}
    return result


def _same_optional_pair(
    first: tuple[float, float] | None,
    second: tuple[float, float] | None,
    *,
    compare: Callable[[tuple[float, float], tuple[float, float]], bool],
) -> bool:
    if first is None or second is None:
        return first is second
    return compare(first, second)


def validate_companion(primary: QpiScan, companion: QpiScan, *, context: ReaderErrorContext) -> None:
    """Require an ordinary 8-bit variant to have the same channel and spatial grid."""
    if (
        primary.dtype.kind != "u"
        or companion.dtype.kind != "u"
        or companion.dtype.itemsize != 1
        or primary.dtype.itemsize <= companion.dtype.itemsize
    ):
        raise ReaderFormatError(
            context=context,
            path=companion.tiff.path,
            found=f"extended-range dtype {primary.dtype} and ordinary dtype {companion.dtype}",
            expected="an unsigned extended-range scan with an unsigned 8-bit companion",
        )
    if (
        primary.shape != companion.shape
        or _channel_identity(primary) != _channel_identity(companion)
        or primary.generation != companion.generation
        or primary.slide_id != companion.slide_id
        or not _same_optional_pair(primary.pixel_size_um, companion.pixel_size_um, compare=matching_pixel_sizes)
        or not _same_optional_pair(
            primary.stage_position_um, companion.stage_position_um, compare=matching_stage_positions
        )
    ):
        raise ReaderFormatError(
            context=context,
            path=companion.tiff.path,
            found="an ordinary scan with a different channel or spatial grid",
            expected="same-generation variants with matching channels, XY shape, calibration and stage",
        )


def _channel_identity(scan: QpiScan) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (channel.channel_id, channel.metadata.get("biomarker"), channel.metadata.get("band_name"))
        for channel in scan.channels
    )
