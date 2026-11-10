"""Read the Space Ranger scale factors and the identity attributes of a 10x HDF5 count matrix.

``spatial/scalefactors_json.json`` relates the full-resolution tissue image to
its two downsampled PNGs and gives display diameters in full-resolution
pixels, for example::

    {
        "tissue_hires_scalef": 0.17011142,  # full-resolution px -> hires px
        "tissue_lowres_scalef": 0.051033426,  # full-resolution px -> lowres px
        "spot_diameter_fullres": 89.43834961,  # display diameter of a spot
        "fiducial_diameter_fullres": 144.48502629,
    }

The three named keys are required positive numbers; every key, including any
not listed, is kept unchanged in ``uns["spatial"]``. Space Ranger estimates
the diameters from the fiducial registration and the nominal spot size, not
from the image's pixel size, so they draw spots but do not calibrate the
image in micrometres. The inverse scale factors register the PNGs to the
full-resolution frame. See
https://www.10xgenomics.com/support/software/space-ranger/latest/analysis/outputs/spatial-outputs

A 10x HDF5 count matrix carries the run's identity as root attributes::

    library_ids = [b"V1_Adult_Mouse_Brain"]  # exactly one; optional
    chemistry_description = b"Spatial 3' v1"  # optional
    software_version = b"spaceranger-2.1.0"  # optional

Text may be stored as fixed- or variable-length strings and must be nonempty
UTF-8 (shared ``decode_hdf5_text``). The software version normalized to
``<major>.<minor>.<patch>`` (``2.1.0``) is the detected format version; other
spellings are kept as text but give no format version. A MEX count directory
has no such attributes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError, add_reader_context
from spatialdata_io.readers._utils.metadata import decode_hdf5_text, read_json_object

if TYPE_CHECKING:
    from spatialdata_io.readers._utils.paths import ArtifactFile

HIRES_SCALE = "tissue_hires_scalef"
LOWRES_SCALE = "tissue_lowres_scalef"
SPOT_DIAMETER = "spot_diameter_fullres"

_SOFTWARE_VERSION = re.compile(r"(?:spaceranger-)?(\d+\.\d+\.\d+)")


@dataclass(frozen=True, slots=True)
class ScaleFactors:
    """Validated scale factors plus the complete owned vendor mapping."""

    hires: float
    lowres: float
    spot_diameter: float
    values: dict[str, object]


@dataclass(frozen=True, slots=True)
class CountMetadata:
    """Identity attributes copied from a closed 10x HDF5 count matrix."""

    library_id: str | None
    chemistry_description: str | None
    software_version: str | None

    @property
    def format_version(self) -> str | None:
        """Return the Space Ranger version as ``<major>.<minor>.<patch>``, if recognized."""
        if self.software_version is None:
            return None
        match = _SOFTWARE_VERSION.fullmatch(self.software_version)
        return None if match is None else match[1]


def _positive_number(values: dict[str, object], key: str, *, path: ArtifactFile, context: ReaderErrorContext) -> float:
    """Return one required positive JSON number as float."""
    value = values.get(key)
    # Strict JSON numbers are finite and within the float range.
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ReaderFormatError(
            context=context, found=f"{key!r}={value!r}", expected=f"a positive numeric {key!r}", path=path
        )
    return float(value)


def read_scale_factors(path: ArtifactFile, *, version: str | None) -> ScaleFactors:
    """Read and validate the Space Ranger scale-factor object."""
    context = ReaderErrorContext("visium", "scale factors", version)
    values = read_json_object(path, context=context)
    hires = _positive_number(values, HIRES_SCALE, path=path, context=context)
    lowres = _positive_number(values, LOWRES_SCALE, path=path, context=context)
    spot_diameter = _positive_number(values, SPOT_DIAMETER, path=path, context=context)
    return ScaleFactors(hires=hires, lowres=lowres, spot_diameter=spot_diameter, values=values)


def _library_id(attributes: h5py.AttributeManager, *, path: ArtifactFile, context: ReaderErrorContext) -> str | None:
    """Return the single library ID of single-library ``spaceranger count`` output."""
    if "library_ids" not in attributes:
        return None

    raw: object = attributes["library_ids"]
    if isinstance(raw, np.ndarray):
        if raw.size != 1:
            raise ReaderFormatError(
                context=context,
                found=f"{raw.size} library IDs",
                expected="exactly one library ID from single-library spaceranger count output",
                path=path,
            )
        raw = raw.reshape(-1)[0]
    return decode_hdf5_text(raw, attribute="library_ids", context=context, path=path)


def read_count_metadata(path: ArtifactFile) -> CountMetadata:
    """Copy the identity attributes of a 10x HDF5 count matrix and close it."""
    context = ReaderErrorContext("visium", "count metadata")
    try:
        with h5py.File(path, mode="r") as source:
            attributes = source.attrs
            chemistry, software = (
                None
                if name not in attributes
                else decode_hdf5_text(attributes[name], attribute=name, context=context, path=path)
                for name in ("chemistry_description", "software_version")
            )
            return CountMetadata(_library_id(attributes, path=path, context=context), chemistry, software)
    except OSError as error:
        add_reader_context(error, context=context, path=path)
        raise
