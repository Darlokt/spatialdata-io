"""Integration tests for closed TIFF and OME metadata inspection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from tifffile.zarr import ZarrTiffStore

from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderErrorContext
from spatialdata_io.readers._utils.image import inspect_tiff, read_tiff
from spatialdata_io.readers._utils.paths import ArtifactFile
from tests.integration.readers._utils.image.conftest import TiffOptions

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture

    from tests.integration.readers._utils.image.conftest import TiffFactory


_RASTER_CONTEXT = ReaderErrorContext("test", "raster")


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

        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)

        level = metadata.series[0].levels[0]
        assert segment_spy.call_count == 0
        assert metadata.path == path.resolve()
        assert metadata.ome is None
        assert level.shape == cyx_data.shape
        assert level.dtype == cyx_data.dtype
        assert level.axes == ("C", "Y", "X")
        assert level.native_chunks == expected_chunks

    def test_closes_successful_inspection_resources(
        self,
        cyx_data: NDArray[np.uint16],
        write_tiff: TiffFactory,
        mocker: MockerFixture,
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(tile=(16, 16)))
        original_file_close = tifffile.FileHandle.close
        original_store_close = ZarrTiffStore.close
        closed_handles: list[tifffile.FileHandle] = []
        closed_stores: list[ZarrTiffStore] = []

        def tracked_file_close(handle: tifffile.FileHandle) -> None:
            original_file_close(handle)
            closed_handles.append(handle)

        def tracked_store_close(store: ZarrTiffStore) -> None:
            original_store_close(store)
            closed_stores.append(store)

        mocker.patch.object(tifffile.FileHandle, "close", new=tracked_file_close)
        mocker.patch.object(ZarrTiffStore, "close", new=tracked_store_close)

        inspect_tiff(path, context=_RASTER_CONTEXT)

        assert len(closed_handles) == 1
        assert closed_handles[0].closed
        assert len(closed_stores) == 1

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

        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)

        assert metadata.series[0].levels[0].shape == cyx_data.shape

    def test_inspects_tiff_content_without_conventional_suffix(
        self, cyx_data: NDArray[np.uint16], write_tiff: TiffFactory
    ) -> None:
        path = write_tiff(cyx_data, TiffOptions(name="image.data"))

        metadata = inspect_tiff(path, context=_RASTER_CONTEXT)

        assert metadata.series[0].levels[0].shape == cyx_data.shape

    def test_inspects_multiple_series_in_file_order(self, tmp_path: Path) -> None:
        path = tmp_path / "series.tif"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

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

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

        assert tuple(level.shape for level in metadata.series[0].levels) == (
            base.shape,
            reduced.shape,
        )

    def test_discovers_linked_ome_paths(self, linked_ome_tiff: tuple[Path, NDArray[np.uint16]]) -> None:
        path, data = linked_ome_tiff

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

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
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    def test_rejects_symlinked_source_without_linked_files_beside_its_target(
        self,
        linked_ome_tiff: tuple[Path, NDArray[np.uint16]],
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        # tifffile reads linked files beside the real path and would zero-fill missing ones.
        path, _ = linked_ome_tiff
        target = tmp_path_factory.mktemp("object-store") / "0123abcd"
        path.replace(target)
        path.symlink_to(target)

        with pytest.raises(RasterFormatError, match="non-file TIFF source"):
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    def test_resolves_renamed_single_file_ome_by_its_own_uuid(self, tmp_path: Path) -> None:
        # Public releases rename exported files; FileName keeps the export name.
        uuid = "urn:uuid:33333333-3333-4333-8333-333333333333"
        xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="{uuid}">
  <Image ID="Image:0"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16"
    SizeX="5" SizeY="4" SizeZ="1" SizeC="1" SizeT="1">
    <Channel ID="Channel:0:0" SamplesPerPixel="1"/>
    <TiffData IFD="0" PlaneCount="1"><UUID FileName="export-name.tif">{uuid}</UUID></TiffData>
  </Pixels></Image>
</OME>"""
        data = np.arange(20, dtype=np.uint16).reshape(4, 5)
        path = tmp_path / "release-name.ome.tif"
        tifffile.imwrite(path, data, description=xml, metadata=None)

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

        assert metadata.series[0].levels[0].shape == (4, 5)
        np.testing.assert_array_equal(
            read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT).data.compute(), data
        )

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

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

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

        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

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
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    def test_rejects_linked_ome_path_outside_source_directory(
        self,
        linked_ome_tiff: tuple[Path, NDArray[np.uint16]],
    ) -> None:
        path, _ = linked_ome_tiff
        second_path = path.with_name("second.ome.tif")
        outside_path = path.parent.parent / "outside.tif"
        second_path.replace(outside_path)
        ome_xml = tifffile.tiffcomment(path)
        assert isinstance(ome_xml, str)
        tifffile.tiffcomment(
            path,
            ome_xml.replace("second.ome.tif", "../outside.tif").encode(),
        )

        with pytest.raises(RasterFormatError, match="escapes the TIFF source directory"):
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    @pytest.mark.parametrize(
        ("first_mapping", "depth", "second_ifd", "overlaps"),
        [
            ('IFD="0" PlaneCount="2"', 2, 1, True),
            ('IFD="0" PlaneCount="2"', 2, 2, False),
            ("", 2, 1, True),
            ("", 2, 2, False),
            ('IFD="0"', 1, 1, False),
        ],
    )
    def test_effective_ifd_ranges(
        self, tmp_path: Path, first_mapping: str, depth: int, second_ifd: int, *, overlaps: bool
    ) -> None:
        path = tmp_path / "planes.ome.tif"
        images = "".join(
            f'<Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYZCT" Type="uint8" '
            f'SizeX="4" SizeY="3" SizeZ="{z}" SizeC="1" SizeT="1">'
            f'<Channel ID="Channel:{i}:0" SamplesPerPixel="1"/><TiffData {mapping}/></Pixels></Image>'
            for i, (z, mapping) in enumerate(((depth, first_mapping), (1, f'IFD="{second_ifd}"')))
        )
        xml = f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">{images}</OME>'
        with tifffile.TiffWriter(path) as writer:
            for plane in range(3):
                writer.write(
                    np.full((3, 4), plane, dtype=np.uint8), description=xml if plane == 0 else None, metadata=None
                )
        if overlaps:
            with pytest.raises(RasterFormatError, match="reuses TIFF IFD"):
                inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        else:
            metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
            assert len(metadata.series) == 2
            raster = read_tiff(metadata, series=1, level=0, context=_RASTER_CONTEXT)
            np.testing.assert_array_equal(raster.data.compute(), np.full((3, 4), second_ifd))

    def test_omitted_count_stops_at_remaining_logical_rgb_planes(self, tmp_path: Path) -> None:
        # Three packed RGB samples occupy one IFD per timepoint. From FirstT=1
        # only IFD 0 remains in the first image; IFD 1 belongs to the second.
        images = "".join(
            f'<Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYCZT" Type="uint8" '
            f'SizeX="4" SizeY="3" SizeZ="1" SizeC="3" SizeT="{times}">'
            f'<Channel ID="Channel:{i}:0" SamplesPerPixel="3"/>{blocks}</Pixels></Image>'
            for i, (times, blocks) in enumerate(
                ((2, '<TiffData IFD="3"/><TiffData FirstT="1"/>'), (1, '<TiffData IFD="1"/>'))
            )
        )
        path = tmp_path / "rgb.ome.tif"
        xml = f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">{images}</OME>'
        with tifffile.TiffWriter(path) as writer:
            for plane in range(4):
                writer.write(
                    np.full((3, 4, 3), plane, dtype=np.uint8),
                    photometric="rgb",
                    description=xml if plane == 0 else None,
                    metadata=None,
                )
        metadata = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        first = read_tiff(metadata, series=0, level=0, context=_RASTER_CONTEXT)
        np.testing.assert_array_equal(first.data[:, 0, 0, 0].compute(), [3, 0])
        second = read_tiff(metadata, series=1, level=0, context=_RASTER_CONTEXT)
        np.testing.assert_array_equal(second.data.compute(), np.ones((3, 4, 3), dtype=np.uint8))

    def test_conflicting_declared_range_cannot_be_overwritten(self, tmp_path: Path) -> None:
        # A decoder can replace the first mapping for image 0 with its second one.
        # The earlier block still explicitly claims IFD 1, also used by image 1.
        images = "".join(
            f'<Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYZCT" Type="uint8" '
            f'SizeX="4" SizeY="3" SizeZ="{z}" SizeC="1" SizeT="1"><Channel ID="Channel:{i}:0"/>'
            f"{blocks}</Pixels></Image>"
            for i, (z, blocks) in enumerate(
                (
                    (2, '<TiffData IFD="0" PlaneCount="2"/><TiffData IFD="2" PlaneCount="2"/>'),
                    (1, '<TiffData IFD="1"/>'),
                )
            )
        )
        path = tmp_path / "overwritten.ome.tif"
        xml = f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">{images}</OME>'
        with tifffile.TiffWriter(path) as writer:
            for plane in range(4):
                writer.write(
                    np.full((3, 4), plane, dtype=np.uint8), description=xml if plane == 0 else None, metadata=None
                )
        with pytest.raises(RasterFormatError, match="reuses TIFF IFD"):
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    @pytest.mark.parametrize(
        ("blocks", "error"),
        [
            ('<TiffData IFD="0" FirstZ="2"/>', "FirstZ"),
            ('<TiffData IFD="7"/>', "only 2 IFDs"),
            ('<TiffData IFD="1" PlaneCount="2"/>', "only 2 IFDs"),
            ('<TiffData IFD="0"/><TiffData IFD="1"/>', "logical plane"),
            ('<TiffData IFD="0"/><TiffData FirstZ="1" IFD="0"/>', "reuses TIFF IFD"),
            ('<TiffData IFD="0" PlaneCount="2"/><TiffData FirstZ="1" IFD="1"/>', None),
        ],
    )
    def test_logical_plane_identity_within_one_image(self, tmp_path: Path, blocks: str, error: str | None) -> None:
        path = ArtifactFile(tmp_path / "planes.ome.tiff")
        xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
            '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8" SizeX="4" SizeY="3" SizeZ="2" SizeC="1" '
            f'SizeT="1"><Channel ID="Channel:0:0"/>{blocks}</Pixels></Image></OME>'
        )
        with tifffile.TiffWriter(path) as writer:
            for index in range(2):
                writer.write(
                    np.full((3, 4), index, dtype=np.uint8), description=xml if index == 0 else None, metadata=None
                )
        if error is not None:
            with pytest.raises(RasterFormatError, match=error):
                inspect_tiff(path, context=_RASTER_CONTEXT)
        else:
            raster = read_tiff(inspect_tiff(path, context=_RASTER_CONTEXT), series=0, level=0, context=_RASTER_CONTEXT)
            np.testing.assert_array_equal(raster.data[:, 0, 0].compute(), [0, 1])

    @pytest.mark.parametrize(
        ("references", "ifds", "error"),
        [
            (("first.tif", "second.tif"), (0, 0), "conflicting file names"),
            (("first.tif", None), (0, 0), "reuses TIFF IFD"),
            ((None, None), (0, 1), "FileName"),
            (("first.tif", None), (0, 1), None),
        ],
    )
    def test_external_uuid_identifies_one_file(
        self, tmp_path: Path, references: tuple[str | None, str | None], ifds: tuple[int, int], error: str | None
    ) -> None:
        root = ArtifactFile(tmp_path / "root.tif")
        external_uuid = "urn:uuid:22222222-2222-4222-8222-222222222222"
        images = []
        for image, (name, ifd) in enumerate(zip(references, ifds, strict=True)):
            attribute = "" if name is None else f' FileName="{name}"'
            images.append(
                f'<Image ID="Image:{image}"><Pixels ID="Pixels:{image}" DimensionOrder="XYZCT" Type="uint8" '
                f'SizeX="4" SizeY="3" SizeZ="1" SizeC="1" SizeT="1"><Channel ID="Channel:{image}:0"/>'
                f'<TiffData IFD="{ifd}"><UUID{attribute}>{external_uuid}</UUID></TiffData></Pixels></Image>'
            )
        xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
            f'UUID="urn:uuid:11111111-1111-4111-8111-111111111111">{"".join(images)}</OME>'
        )
        tifffile.imwrite(root, np.zeros((3, 4), dtype=np.uint8), description=xml, metadata=None)
        for name, first in (("first.tif", 11), ("second.tif", 73)):
            with tifffile.TiffWriter(tmp_path / name) as writer:
                for value in (first, first + 1):
                    writer.write(np.full((3, 4), value, dtype=np.uint8), metadata=None)
        if error is not None:
            with pytest.raises(RasterFormatError, match=error):
                inspect_tiff(root, context=_RASTER_CONTEXT)
        else:
            metadata = inspect_tiff(root, context=_RASTER_CONTEXT)
            for series, value in enumerate((11, 12)):
                raster = read_tiff(metadata, series=series, level=0, context=_RASTER_CONTEXT)
                np.testing.assert_array_equal(raster.data.compute(), np.full((3, 4), value))

    def test_explicit_zero_plane_count_cannot_decode_an_entire_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty_mapping.ome.tif"
        xml = (
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0">'
            '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint8" SizeX="4" SizeY="3" '
            'SizeZ="1" SizeC="1" SizeT="1"><Channel ID="Channel:0:0"/><TiffData PlaneCount="0"/>'
            "</Pixels></Image></OME>"
        )
        tifffile.imwrite(path, np.ones((3, 4), dtype=np.uint8), description=xml, metadata=None)
        with pytest.raises(RasterFormatError, match="PlaneCount"):
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

    def test_equal_ifd_numbers_in_distinct_files(self, tmp_path: Path) -> None:
        first, second = tmp_path / "first.tif", tmp_path / "second.tif"
        first_uuid = "urn:uuid:11111111-1111-4111-8111-111111111111"
        second_uuid = "urn:uuid:22222222-2222-4222-8222-222222222222"
        images = "".join(
            f'<Image ID="Image:{i}"><Pixels ID="Pixels:{i}" DimensionOrder="XYZCT" Type="uint8" '
            f'SizeX="4" SizeY="3" SizeZ="1" SizeC="1" SizeT="1"><Channel ID="Channel:{i}:0" SamplesPerPixel="1"/>'
            f'<TiffData><UUID FileName="{path.name}">{uuid}</UUID></TiffData></Pixels></Image>'
            for i, (path, uuid) in enumerate(((first, first_uuid), (second, second_uuid)))
        )
        xml = f'<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" UUID="{first_uuid}">{images}</OME>'
        tifffile.imwrite(first, np.zeros((3, 4), dtype=np.uint8), description=xml, metadata=None)
        tifffile.imwrite(second, np.ones((3, 4), dtype=np.uint8), metadata=None)

        metadata = inspect_tiff(ArtifactFile(first), context=_RASTER_CONTEXT)
        for index in range(2):
            raster = read_tiff(metadata, series=index, level=0, context=_RASTER_CONTEXT)
            np.testing.assert_array_equal(raster.data.compute(), np.full((3, 4), index))

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
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

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
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
        assert close_spy.call_count >= 1
