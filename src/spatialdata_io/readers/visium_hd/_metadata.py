"""Read Space Ranger HD identity, calibration and feature-slice metadata.

Every 10x HDF5 count matrix and the feature slice carry the run's identity as
root attributes::

    software_version = "spaceranger-4.0.1"  # optional; "4.0.1" also accepted
    library_ids = [b"<library ID>"]  # optional; exactly one when present

Text may be stored as fixed- or variable-length strings and must be nonempty
UTF-8 (shared ``decode_hdf5_text``). ``spaceranger-X.Y.Z`` or ``X.Y.Z`` is
normalized to ``X.Y.Z``, the detected format version recorded in provenance
and error contexts; other spellings (``spaceranger-4.0.1-rc1``, ``4.1``) and a
missing attribute give no format version, as in the classic Visium reader.
Nothing branches on the version: the layout is recognized from the artifacts
present (``hd_feature_slice.h5`` in 3.0, ``segmented_outputs/`` since 4.0,
centroid properties since 4.1). The reader is tested with Space Ranger 3.0,
3.1, 4.0 and 4.1; another release is read with one ``UserWarning`` that it is
untested.

Consumed sources must come from one run. Every detected format version must be
the same full version (sources without one do not take part), and at most one
distinct library ID may occur (sources without one do not take part either).

Each bin directory has ``spatial/scalefactors_json.json``; at 8 µm in the
public 4.0.1 tiny mouse brain example::

    {
        "spot_diameter_fullres": 7.975055448344061,  # bin side in full-resolution pixels
        "bin_size_um": 8.0,
        "microns_per_pixel": 1.003127821720803,  # full-resolution pixel size
        "tissue_lowres_scalef": 0.1,  # full-resolution px -> lowres px
        "fiducial_diameter_fullres": 328.9710372441925,
        "tissue_hires_scalef": 1.0,  # full-resolution px -> hires px
        "regist_target_img_scalef": 1.0,
    }

The five named keys are required, finite and positive; ``bin_size_um`` must
equal the directory's size and ``bin_size_um / spot_diameter_fullres`` must
equal ``microns_per_pixel`` (relative tolerance 1e-12), so a file copied from
another bin size is rejected. ``microns_per_pixel`` is Space Ranger's nominal
estimate; the returned frame is calibrated by the manufactured 2 µm grid
instead (see ``_transforms``) and keeps it only as a cross-check.

``feature_slice.h5`` (``hd_feature_slice.h5`` in 3.0) holds a JSON
``metadata_json`` attribute, of which the reader consumes::

    {
        "nrows": 3350,
        "ncols": 3350,  # 2 µm grid dimensions
        "spot_pitch": 2.0,  # µm, must be 2
        "transform_matrices": {  # 3 x 3 row-major, each pair nullable
            "spot_colrow_to_microscope_colrow": [...],
            "microscope_colrow_to_spot_colrow": [...],
            "spot_colrow_to_cytassist_colrow": [...],
            "cytassist_colrow_to_spot_colrow": [...],
        },
    }

Each matrix pair is two finite invertible mutual inverses or two nulls. Only
the forward CytAssist matrix is used (to place the CytAssist image); the
microscope registration is refitted from the positions. Other keys
(``sample_id``, ``hd_layout_json``, ...) are ignored. See
https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context, warn_user
from spatialdata_io.readers._utils.metadata import decode_hdf5_text, parse_json_object, read_json_object
from spatialdata_io.readers.visium_hd._transforms import SPOT_PITCH_UM

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from spatialdata_io.readers._utils.paths import ArtifactFile

_VERSION = re.compile(r"(?:spaceranger-)?(\d+\.\d+\.\d+)")
# Releases whose outputs are covered by fixtures; others read with a warning.
_TESTED_RELEASES = ((3, 0), (3, 1), (4, 0), (4, 1))


@dataclass(frozen=True, slots=True)
class Identity:
    """Optional source library identity and normalized Space Ranger version."""

    format_version: str | None
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


def _positive(values: dict[str, object], name: str, *, path: ArtifactFile, context: ReaderErrorContext) -> float:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=f"{name}={value!r}", expected="a finite positive number"
        )
    return float(value)


def read_bin_scale_factors(path: ArtifactFile, size_um: int, *, version: str | None) -> ScaleFactors:
    """Read raster scales and validate the expected bin's nominal calibration."""
    context = ReaderErrorContext("visium_hd", "scale factors", version)
    values = read_json_object(path, context=context)
    factors = ScaleFactors(
        hires=_positive(values, "tissue_hires_scalef", path=path, context=context),
        lowres=_positive(values, "tissue_lowres_scalef", path=path, context=context),
        microns_per_pixel=_positive(values, "microns_per_pixel", path=path, context=context),
    )
    diameter = _positive(values, "spot_diameter_fullres", path=path, context=context)
    bin_size = _positive(values, "bin_size_um", path=path, context=context)
    if bin_size != size_um:
        raise ReaderFormatError(
            context=context, path=path, found=f"bin_size_um={bin_size!r}", expected=f"bin_size_um={size_um}"
        )
    if not math.isclose(size_um / diameter, factors.microns_per_pixel, rel_tol=1e-12):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="inconsistent bin pitch, diameter and microns_per_pixel",
            expected="bin_size_um / spot_diameter_fullres = microns_per_pixel",
        )

    return factors


def read_identity(path: ArtifactFile) -> Identity:
    """Read a source's normalized Space Ranger version and library ID, either possibly absent."""
    context = ReaderErrorContext("visium_hd", "count identity")
    try:
        with h5py.File(path, "r") as source:
            version: str | None = None
            if "software_version" in source.attrs:
                text = decode_hdf5_text(
                    source.attrs["software_version"], attribute="software_version", context=context, path=path
                )
                match = _VERSION.fullmatch(text)
                if match is not None:
                    version = match[1]
                    context = ReaderErrorContext("visium_hd", "count identity", version)

            raw: object = source.attrs.get("library_ids")
            if isinstance(raw, np.ndarray) and raw.ndim:
                if raw.size != 1:
                    raise ReaderFormatError(
                        context=context, path=path, found=f"{raw.size} library IDs", expected="exactly one library ID"
                    )
                raw = raw.reshape(-1)[0]
            library_id = (
                None if raw is None else decode_hdf5_text(raw, attribute="library_ids", context=context, path=path)
            )
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise

    return Identity(version, library_id)


def reconcile_identities(paths: Sequence[ArtifactFile]) -> Identity:
    """Read each distinct consumed source once and reconcile its export identity.

    Sources without a detected version or library ID do not constrain that
    field; the known values must agree. An untested release warns once.
    """
    if not paths:
        raise ReaderFormatError(
            context=ReaderErrorContext("visium_hd", "count identity"),
            found="no selected identity source",
            expected="at least one selected identity-bearing artifact",
        )

    version: str | None = None
    library_id: str | None = None
    for path in dict.fromkeys(paths):
        candidate = read_identity(path)
        context = ReaderErrorContext("visium_hd", "count identity", candidate.format_version or version)
        if candidate.format_version is not None:
            if version is not None and candidate.format_version != version:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"software_version={candidate.format_version!r}",
                    expected=f"software_version={version!r} shared by all consumed sources",
                )
            version = candidate.format_version
        if candidate.library_id is not None:
            if library_id is not None and candidate.library_id != library_id:
                raise ReaderFormatError(
                    context=context,
                    path=path,
                    found=f"library_id={candidate.library_id!r}",
                    expected=f"library_id={library_id!r} shared by all known library IDs",
                )
            library_id = candidate.library_id

    if version is not None and tuple(map(int, version.split(".")[:2])) not in _TESTED_RELEASES:
        tested = ", ".join(f"{major}.{minor}" for major, minor in _TESTED_RELEASES)
        warn_user(
            f"visium_hd: Space Ranger {version} is untested (tested releases: {tested}); "
            "its layout is read from the artifacts present."
        )
    return Identity(version, library_id)


def _matrix(value: object, *, path: ArtifactFile, name: str, context: ReaderErrorContext) -> NDArray[np.float64] | None:
    if value is None:
        return None

    numeric = np.asarray(value, dtype=object)
    if numeric.shape not in {(9,), (3, 3)} or any(
        isinstance(element, bool) or not isinstance(element, int | float) for element in numeric.flat
    ):
        raise ReaderFormatError(
            context=context, path=path, found=f"malformed {name}", expected="a numeric 3 by 3 matrix or null"
        )

    matrix = numeric.astype(np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ReaderFormatError(
            context=context, path=path, found=f"nonfinite or singular {name}", expected="a finite invertible matrix"
        )

    matrix.flags.writeable = False
    return matrix


def _dimension(values: dict[str, object], name: str, *, path: ArtifactFile, context: ReaderErrorContext) -> int:
    value = values.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=f"{name}={value!r}", expected="a positive integer grid dimension"
        )

    return value


def _matrix_pair(
    values: dict[str, object], frame: str, *, path: ArtifactFile, context: ReaderErrorContext
) -> NDArray[np.float64] | None:
    forward = f"spot_colrow_to_{frame}_colrow"
    inverse = f"{frame}_colrow_to_spot_colrow"
    if forward not in values or inverse not in values:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"incomplete {frame} matrix pair",
            expected="both forward and inverse keys",
        )

    first = _matrix(values[forward], path=path, name=forward, context=context)
    second = _matrix(values[inverse], path=path, name=inverse, context=context)
    if (first is None) != (second is None):
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"partly null {frame} matrix pair",
            expected="two matrices or two null values",
        )
    if first is not None and second is not None:
        product = first @ second
        if product[2, 2] == 0 or not np.allclose(product / product[2, 2], np.eye(3), rtol=1e-6, atol=1e-6):
            raise ReaderFormatError(
                context=context, path=path, found=f"inconsistent {frame} matrix pair", expected="mutual inverses"
            )
    return first


def read_source_metadata(path: ArtifactFile, *, version: str | None) -> SourceMetadata:
    """Validate feature metadata, retaining dimensions and the forward CytAssist map."""
    context = ReaderErrorContext("visium_hd", "feature slice metadata", version)
    try:
        with h5py.File(path, "r") as source:
            text = decode_hdf5_text(
                source.attrs.get("metadata_json"), attribute="metadata_json", context=context, path=path
            )
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
    values = parse_json_object(text, context=context, path=path)

    matrices = values.get("transform_matrices")
    if not isinstance(matrices, dict):
        raise ReaderFormatError(
            context=context,
            path=path,
            found="missing transform_matrices object",
            expected="nullable forward/inverse matrix pairs",
        )

    _matrix_pair(matrices, "microscope", path=path, context=context)
    cytassist = _matrix_pair(matrices, "cytassist", path=path, context=context)

    pitch = _positive(values, "spot_pitch", path=path, context=context)
    if pitch != SPOT_PITCH_UM:
        raise ReaderFormatError(
            context=context,
            path=path,
            found=f"spot_pitch={pitch!r}",
            expected="the supported 2 micrometre HD grid pitch",
        )

    return SourceMetadata(
        _dimension(values, "nrows", path=path, context=context),
        _dimension(values, "ncols", path=path, context=context),
        cytassist,
    )
