"""The registration conditions for QPI bit-depth companions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.phenocycler._images import inspect_scan
from spatialdata_io.readers.phenocycler._transforms import validate_companion
from tests.integration.readers.phenocycler._fixtures import write_qptiff

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")


class TestValidateCompanion:
    @pytest.mark.parametrize(
        ("dtype", "names", "resolution_unit", "failure"),
        [
            ("uint16", ("DAPI", "CD3"), "CENTIMETER", "unsigned 8-bit"),
            ("uint8", ("DAPI", "ATTO 550"), "CENTIMETER", "different channel or spatial grid"),
            ("uint8", ("DAPI", "CD3"), "INCH", "different channel or spatial grid"),
        ],
    )
    def test_requires_measured_dtype_channel_and_grid(
        self, tmp_path: Path, dtype: str, names: tuple[str, str], resolution_unit: str, failure: str
    ) -> None:
        high = tmp_path / "x.er.qptiff"
        low = tmp_path / "x.qptiff"
        write_qptiff(high)
        write_qptiff(low, dtype=dtype, names=names, resolution_unit=resolution_unit)
        primary = inspect_scan(require_file(high, context=_CONTEXT), context=_CONTEXT)
        companion = inspect_scan(require_file(low, context=_CONTEXT), context=_CONTEXT)
        with pytest.raises(ReaderFormatError, match=failure):
            validate_companion(primary, companion, context=_CONTEXT)
