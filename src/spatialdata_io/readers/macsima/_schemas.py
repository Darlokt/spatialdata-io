"""MACSiQView annotation vocabularies, units, and the channel-table contract.

MACSiQView writes one OME-TIFF per channel and describes the acquisition in
OME map annotations whose keys differ between software majors 0 and 1. This
module declares the accepted source vocabularies and units, and the column
types of the metadata-only channel table. Key selection and fallback order
belong to the version parsers in ``_metadata.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

import pandas as pd

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from pandas.api.extensions import ExtensionArray

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


@dataclass(frozen=True, slots=True)
class ChannelMetadata:
    """Acquisition annotations of one channel, normalized across schema versions.

    ``name`` includes the ``_background`` suffix for bleach images. Cycles and ROI
    identifiers are nonnegative uint64 values and ``exposure_ms`` is in milliseconds.
    """

    name: str
    cycle: int
    image_type: ImageType
    well: str | None
    roi: int | None
    fluorophore: str | None
    clone: str | None
    exposure_ms: float | None


def channel_var_columns(rows: Sequence[ChannelMetadata]) -> dict[str, ExtensionArray]:
    """Construct the channel-table columns in output order without dtype inference."""
    return {
        "name": pd.array([row.name for row in rows], dtype="string"),
        "cycle": pd.array([row.cycle for row in rows], dtype="uint64"),
        "imagetype": pd.array([row.image_type for row in rows], dtype="string"),
        "well": pd.array([row.well for row in rows], dtype="string"),
        # Nullable integers keep wide ROI IDs exact in the presence of missing values.
        "ROI": pd.array([row.roi for row in rows], dtype="UInt64"),
        "fluorophore": pd.array([row.fluorophore for row in rows], dtype="string"),
        "clone": pd.array([row.clone for row in rows], dtype="string"),
        "exposure": pd.array([row.exposure_ms for row in rows], dtype="float64"),
    }


def normalize_image_type(value: str, *, major: SchemaMajor, context: ReaderErrorContext, path: Path) -> ImageType:
    """Map one version-specific acquisition type to the cross-version vocabulary."""
    vocabulary = _IMAGE_TYPES[major]
    if value not in vocabulary:
        raise ReaderFormatError(
            context=context, path=path, found=f"image type {value!r}", expected=f"one of {sorted(vocabulary)}"
        )
    return vocabulary[value]
