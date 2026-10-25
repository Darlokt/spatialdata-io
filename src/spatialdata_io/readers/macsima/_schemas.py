"""MACSiQView annotation vocabularies, units, and the channel-table contract.

MACSiQView writes one OME-TIFF per channel and describes the acquisition in
OME map annotations whose keys differ between software majors 0 and 1. This
module declares the accepted source vocabularies and units, and the column
types of the metadata-only channel table. Key selection and fallback order
belong to the version parsers in ``_metadata.py``.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from ome_types.model import UnitsLength

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

type ImageType = Literal["stain", "bleach", "autofluorescence"]
type SchemaMajor = Literal[0, 1]

# Source vocabularies of the acquisition type ("MICS cycle type" in v0,
# "ScanType" in v1), normalized to one cross-version vocabulary.
_IMAGE_TYPES: Mapping[SchemaMajor, Mapping[str, ImageType]] = MappingProxyType(
    {
        0: MappingProxyType({"AntigenCycle": "stain", "BleachCycle": "bleach"}),
        1: MappingProxyType({"S": "stain", "B": "bleach", "AF": "autofluorescence"}),
    }
)

# MACSiQView writes exposure times in milliseconds; an absent unit annotation
# is read as milliseconds.
EXPOSURE_UNIT = "ms"

# Divisors convert OME physical pixel sizes to micrometers in one correctly
# rounded operation (170 nm becomes exactly the float nearest to 0.17 µm).
_MICROMETER_DIVISORS: Mapping[UnitsLength, int] = MappingProxyType(
    {UnitsLength.MICROMETER: 1, UnitsLength.NANOMETER: 1000}
)

# Columns of the channel table's ``var`` in output order. ``name`` holds the
# marker (bleach images carry a ``_background`` suffix), ``exposure`` is in
# milliseconds and NaN when absent; other optional values are missing as NA.
CHANNEL_VAR_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "name": "string",
        "cycle": "int64",
        "imagetype": "string",
        "well": "string",
        "ROI": "Int64",
        "fluorophore": "string",
        "clone": "string",
        "exposure": "float64",
    }
)


def normalize_image_type(value: str, *, major: SchemaMajor, context: ReaderErrorContext, path: Path) -> ImageType:
    """Map one version-specific acquisition type to the cross-version vocabulary."""
    vocabulary = _IMAGE_TYPES[major]
    if value not in vocabulary:
        raise ReaderFormatError(
            context=context, path=path, found=f"image type {value!r}", expected=f"one of {sorted(vocabulary)}"
        )
    return vocabulary[value]


def pixel_size_um(size: float | None, unit: UnitsLength, *, context: ReaderErrorContext, path: Path) -> float:
    """Convert one finite positive OME physical pixel size to micrometers."""
    if unit not in _MICROMETER_DIVISORS:
        raise ReaderFormatError(
            context=context, path=path, found=f"physical size unit {unit.value!r}", expected="µm or nm"
        )
    if size is None or not math.isfinite(size) or size <= 0:
        raise ReaderFormatError(
            context=context, path=path, found=f"physical pixel size {size!r}", expected="a finite positive size"
        )
    return size / _MICROMETER_DIVISORS[unit]
