"""Read the grid pitch for image-only reads from the sample's GEF attributes.

Images carry no physical calibration. When no count or analysis product is
selected, the pitch comes from the ``resolution`` attributes of the sample's
square or cell GEFs in the producer count directories (not in flat exports);
only attributes are read, never count or coordinate arrays. All found pitches
must agree. Without such a GEF the images stay in the ``image_pixels`` frame.
These GEFs only calibrate images, so their schema ``version`` is not read and
an image-only result records no format version.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py

from spatialdata_io.readers._utils.errors import ReaderErrorContext, add_reader_context
from spatialdata_io.readers.stereoseq import _schemas
from spatialdata_io.readers.stereoseq._layout import CellSource

if TYPE_CHECKING:
    from spatialdata_io.readers.stereoseq._layout import SquareSource


def _pitches(handle: h5py.File, source: SquareSource | CellSource) -> tuple[int, ...]:
    path = source.file
    if isinstance(source, CellSource):
        _schemas.require_group(handle, "cellBin", path)
        return (_schemas.integer_attribute(handle.attrs, "resolution", path, "cell resolution", positive=True),)
    group = _schemas.require_group(handle, "geneExp", path)
    bins = [name for name in group if name.startswith("bin") and name[3:].isdigit() and int(name[3:]) > 0]
    if not bins:
        raise _schemas.schema_error(
            path, "image calibration", found="no stored square bins", expected="GEF expression pitch metadata"
        )
    return tuple(
        _schemas.integer_attribute(
            _schemas.require_dataset(_schemas.require_group(group, name, path), "expression", path).attrs,
            "resolution",
            path,
            "expression resolution",
            positive=True,
        )
        for name in sorted(bins)
    )


def read_image_pitch(sources: tuple[SquareSource | CellSource, ...], *, sample_id: str | None) -> int | None:
    """Read and reconcile scalar nm pitches from already associated companions."""
    pitch: int | None = None
    for source in sources:
        try:
            with h5py.File(source.file, "r") as handle:
                _schemas.validate_transcriptomics(handle.attrs, source.file)
                sample = _schemas.sample_id(handle, source.file)
                if sample is not None and sample != sample_id:
                    raise _schemas.schema_error(
                        source.file,
                        "image calibration",
                        found=f"GEF sample {sample!r}",
                        expected=f"selected sample {sample_id!r}",
                    )
                for value in _pitches(handle, source):
                    if pitch is not None and pitch != value:
                        raise _schemas.schema_error(
                            source.file,
                            "image calibration",
                            found=f"conflicting pitch {value} nm",
                            expected=f"common grid pitch {pitch} nm",
                        )
                    pitch = value
        except OSError as error:
            add_reader_context(error, context=ReaderErrorContext("stereoseq", "image calibration"), path=source.file)
            raise
    return pitch
