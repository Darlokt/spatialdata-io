"""Integration tests for persisted reader provenance."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata import SpatialData, read_zarr

from spatialdata_io import __version__
from spatialdata_io.readers._utils.provenance import set_reader_provenance

if TYPE_CHECKING:
    from pathlib import Path


class TestSetReaderProvenance:
    def test_spatialdata_zarr_round_trip(self, tmp_path: Path) -> None:
        sdata = SpatialData(attrs={"unrelated": "preserved"})
        set_reader_provenance(sdata.attrs, reader="xenium", format_version="3.0.1")

        output = tmp_path / "provenance.zarr"
        sdata.write(output)
        restored = read_zarr(output)

        assert restored.attrs == {
            "unrelated": "preserved",
            "spatialdata_io_software_version": __version__,
            "spatialdata_io_reader": "xenium",
            "spatialdata_io_format_version": "3.0.1",
        }
