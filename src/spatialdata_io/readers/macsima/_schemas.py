"""MACSiQView annotation vocabularies, units, and the channel-table contract.

MACSiQView writes one OME-TIFF per channel and describes the acquisition in
OME map annotations whose keys differ between software majors 0 and 1. This
module declares the accepted source vocabularies and units, and the column
types of the metadata-only channel table. Key selection and fallback order
belong to the version parsers in ``_metadata.py``.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

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

# Columns of the channel table's ``var`` in output order. ``name`` holds the
# marker (bleach images carry a ``_background`` suffix), ``exposure`` is in
# milliseconds and NaN when absent; other optional values are missing as NA.
# Cycles and ROI IDs are checked nonnegative uint64 values in the XML parser.
CHANNEL_VAR_DTYPES: Mapping[str, str] = MappingProxyType(
    {
        "name": "string",
        "cycle": "uint64",
        "imagetype": "string",
        "well": "string",
        "ROI": "UInt64",
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
