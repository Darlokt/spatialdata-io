"""Integration tests for closed TIFF and OME metadata inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.image import inspect_tiff
from spatialdata_io.readers._utils.image._exceptions import RasterFormatError
from tests.integration.readers._utils.image.conftest import TiffOptions

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from tests.integration.readers._utils.image.conftest import TiffFactory


class TestInspectTiff:
    """Inspect complete deterministic TIFF structure without decoding segments."""

    @pytest.mark.parametrize(
        "storage",
        [
            (None, None, None, (1, 19, 23)),
            (None, None, 5, (1, 5, 23)),
            ("deflate", (16, 16), None, (1, 16, 16)),
            ("lzw", (16, 16), None, (1, 16, 16)),
        ],
    )
    def test_records_native_geometry_without_decoding(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
        storage: tuple[str | None, tuple[int, int] | None, int | None, tuple[int, ...]],
    ) -> None:
        compression, tile, rowsperstrip, expected_chunks = storage
        path = write_tiff(
            cyx_data,
            TiffOptions(compression=compression, tile=tile, rowsperstrip=rowsperstrip),
        )
        segment_spy = mocker.spy(tifffile.FileHandle, "read_segments")

        metadata = inspect_tiff(path)

        level = metadata.series[0].levels[0]
        assert segment_spy.call_count == 0
        assert metadata.path == path.resolve()
        assert metadata.ome is None
        assert level.shape == cyx_data.shape
        assert level.dtype == cyx_data.dtype
        assert level.axes == ("C", "Y", "X")
        assert level.native_chunks == expected_chunks

    def test_inspects_bigtiff_btf(self, cyx_data: NDArray[np.uint16], write_tiff: TiffFactory) -> None:
        path = write_tiff(
            cyx_data,
            TiffOptions(
                name="image.btf",
                compression="deflate",
                tile=(16, 16),
                bigtiff=True,
            ),
        )

        metadata = inspect_tiff(path)

        assert metadata.series[0].levels[0].shape == cyx_data.shape

    def test_inspects_tiff_content_without_conventional_suffix(
        self, cyx_data: NDArray[np.uint16], write_tiff: TiffFactory
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(name="image.data"))

        metadata = inspect_tiff(path)

        assert metadata.series[0].levels[0].shape == cyx_data.shape

    def test_inspects_multiple_series_in_file_order(self, tmp_path: Path) -> None:
        path = tmp_path / "series.tif"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        metadata = inspect_tiff(path)

        assert tuple(series.levels[0].shape for series in metadata.series) == (
            first.shape,
            second.shape,
        )

    def test_inspects_subifd_pyramid_levels_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "pyramid.tif"
        base = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        reduced = base[::2, ::2]
        with tifffile.TiffWriter(path) as writer:
            writer.write(
                base,
                tile=(16, 16),
                subifds=1,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )
            writer.write(reduced, tile=(16, 16), subfiletype=1, photometric="minisblack")

        metadata = inspect_tiff(path)

        assert tuple(level.shape for level in metadata.series[0].levels) == (
            base.shape,
            reduced.shape,
        )

    def test_discovers_linked_ome_paths(self, linked_ome_tiff: tuple[Path, NDArray[np.uint16]]) -> None:
        path, data = linked_ome_tiff

        metadata = inspect_tiff(path)

        assert metadata.ome is not None
        assert metadata.series[0].levels[0].shape == data.shape
        assert metadata.series[0].levels[0].axes == ("T", "Y", "X")

    def test_rejects_missing_linked_ome_source(
        self,
        linked_ome_tiff: tuple[Path, NDArray[np.uint16]],
    ) -> None:
        path, _ = linked_ome_tiff
        path.with_name("second.ome.tif").unlink()

        with pytest.raises(RasterFormatError, match="non-file TIFF source"):
            inspect_tiff(path)

    def test_preserves_missing_duplicate_channels_and_anisotropic_units(
        self,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "metadata.ome.tif"
        data = np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9)
        tifffile.imwrite(
            path,
            data,
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "CYX",
                "Channel": {"Name": ["DAPI", None, "DAPI"]},
                "PhysicalSizeX": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 750.0,
                "PhysicalSizeYUnit": "nm",
            },
        )
        ome_xml = tifffile.tiffcomment(path)
        assert isinstance(ome_xml, str)
        missing_name_xml = ome_xml.replace('Name="None"', " " * len('Name="None"'))
        tifffile.tiffcomment(path, missing_name_xml.encode())

        metadata = inspect_tiff(path)

        assert metadata.ome is not None
        pixels = metadata.ome.images[0].pixels
        assert tuple(channel.name for channel in pixels.channels) == ("DAPI", None, "DAPI")
        assert pixels.physical_size_x == 0.5
        assert pixels.physical_size_x_unit is not None
        assert pixels.physical_size_x_unit.value == "µm"
        assert pixels.physical_size_y == 750.0
        assert pixels.physical_size_y_unit is not None
        assert pixels.physical_size_y_unit.value == "nm"

    def test_exposes_complete_typed_ome_dimensions_and_calibration(self, tmp_path: Path) -> None:
        path = tmp_path / "xyzct.ome.tif"
        data = np.arange(2 * 3 * 4 * 5, dtype=np.uint16).reshape(2, 3, 4, 5)
        tifffile.imwrite(
            path,
            data,
            ome=True,
            photometric="minisblack",
            metadata={
                "axes": "ZCYX",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.5,
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeZ": 1.5,
                "PhysicalSizeZUnit": "µm",
                "TimeIncrement": 2.0,
                "TimeIncrementUnit": "s",
            },
        )

        metadata = inspect_tiff(path)

        assert metadata.ome is not None
        pixels = metadata.ome.images[0].pixels
        assert (pixels.size_x, pixels.size_y, pixels.size_z, pixels.size_c, pixels.size_t) == (
            5,
            4,
            2,
            3,
            1,
        )
        assert (
            pixels.physical_size_x,
            pixels.physical_size_y,
            pixels.physical_size_z,
        ) == (0.25, 0.5, 1.5)
        assert pixels.physical_size_x_unit is not None
        assert pixels.physical_size_y_unit is not None
        assert pixels.physical_size_z_unit is not None
        assert (
            pixels.physical_size_x_unit.value,
            pixels.physical_size_y_unit.value,
            pixels.physical_size_z_unit.value,
        ) == ("µm", "µm", "µm")
        assert pixels.dimension_order.value == "XYCZT"
        assert pixels.time_increment == 2.0
        assert pixels.time_increment_unit is not None
        assert pixels.time_increment_unit.value == "s"

    def test_rejects_corrupt_linked_ome_source(
        self,
        linked_ome_tiff: tuple[Path, NDArray[np.uint16]],
    ) -> None:
        path, _ = linked_ome_tiff
        path.with_name("second.ome.tif").write_bytes(b"not a TIFF")

        with pytest.raises(RasterFormatError, match="signature"):
            inspect_tiff(path)

    def test_rejects_inconsistent_ome_image_mapping(self, tmp_path: Path) -> None:
        path = tmp_path / "inconsistent.ome.tif"
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8"
    SizeX="4" SizeY="3" SizeZ="1" SizeC="1" SizeT="1">
    <Channel ID="Channel:0:0" SamplesPerPixel="1"/><TiffData PlaneCount="1"/>
  </Pixels></Image>
  <Image ID="Image:1"><Pixels ID="Pixels:1" DimensionOrder="XYZCT" Type="uint8"
    SizeX="4" SizeY="3" SizeZ="1" SizeC="1" SizeT="1">
    <Channel ID="Channel:1:0" SamplesPerPixel="1"/><TiffData PlaneCount="1"/>
  </Pixels></Image>
</OME>"""
        tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint8), description=xml, metadata=None)

        with pytest.raises(RasterFormatError, match="reuses TIFF IFD"):
            inspect_tiff(path)

    def test_closes_tiff_after_malformed_ome_metadata(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "malformed.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((3, 4), dtype=np.uint8),
            description='<?xml version="1.0"?><OME>',
            metadata=None,
        )
        close_spy = mocker.spy(tifffile.FileHandle, "close")

        with pytest.raises(RasterFormatError, match="Cannot parse OME metadata"):
            inspect_tiff(path)
        assert close_spy.call_count >= 1
