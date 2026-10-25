"""One inspection per channel file and exact v0/v1 OME interpretation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.macsima import _metadata
from spatialdata_io.readers.macsima._metadata import ChannelMetadata, read_channel
from tests.integration.readers.macsima._fixtures import (
    V0_CHANNELS,
    V1_CHANNELS,
    ChannelSpec,
    SchemaVersion,
    channel_ome_xml,
    write_channel,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _write_xml(directory: Path, xml: str, shape: tuple[int, ...] = (12, 15)) -> ArtifactFile:
    path = directory / "channel.tif"
    pixels = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    tifffile.imwrite(path, pixels, description=xml.encode(), metadata=None, photometric="minisblack")
    return ArtifactFile(path.resolve())


def _xml(spec: ChannelSpec, version: SchemaVersion = 1, physical_size: tuple[str, str] = ("0.17", "µm")) -> str:
    return channel_ome_xml(spec, version=version, file_name="channel.tif", index=0, physical_size=physical_size)


def _read(directory: Path, spec: ChannelSpec, version: SchemaVersion) -> ChannelMetadata:
    write_channel(directory, spec, version=version, index=0, stored_name="channel.tif")
    source = read_channel(ArtifactFile((directory / "channel.tif").resolve()))
    assert source is not None
    return source.metadata


class TestReadChannel:
    """Interpret one channel file from a single inspection."""

    def test_v0_export(self, tmp_path: Path) -> None:
        write_channel(tmp_path, V0_CHANNELS[0], version=0, index=0, stored_name="channel.tif")
        path = ArtifactFile((tmp_path / "channel.tif").resolve())

        source = read_channel(path)

        assert source is not None
        assert source.tiff.path == path
        assert (source.software_version, source.major) == ("0.15.3", 0)
        assert source.metadata == ChannelMetadata(
            name="CD15",
            cycle=1,
            image_type="stain",
            well="C-1",
            roi=1,
            fluorophore="APC",
            clone="VIMC6",
            exposure_ms=2304.0,
        )
        assert source.pixel_size_um == 0.17
        assert source.offset_yx == (0, 0)
        assert source.shape_yx == (12, 15)
        assert source.dtype == np.dtype(np.uint16)

    def test_v1_export_with_a_dye(self, tmp_path: Path) -> None:
        source = read_channel(_write_xml(tmp_path, _xml(V1_CHANNELS[1])))

        assert source is not None
        assert (source.software_version, source.major) == ("1.1.1", 1)
        # Dyes have an empty biomarker; the zero-padded ROI ID is a number.
        assert source.metadata == ChannelMetadata(
            name="DAPI",
            cycle=1,
            image_type="stain",
            well="D01",
            roi=1,
            fluorophore="DAPI",
            clone=None,
            exposure_ms=51.0,
        )

    @pytest.mark.parametrize(
        ("spec", "version", "name", "image_type"),
        [
            (ChannelSpec("DAPI", 15, "DAPI", scan="B"), 1, "DAPI_background", "bleach"),
            (ChannelSpec("DAPI", 15, "DAPI", scan="B"), 0, "DAPI_background", "bleach"),
            (ChannelSpec("Autofluorescence", 3, "FITC", scan="AF"), 1, "Autofluorescence", "autofluorescence"),
        ],
    )
    def test_acquisition_types(
        self, tmp_path: Path, spec: ChannelSpec, version: SchemaVersion, name: str, image_type: str
    ) -> None:
        metadata = _read(tmp_path, spec, version)

        assert (metadata.name, metadata.image_type) == (name, image_type)

    def test_v0_uses_the_mics_cycle_id_without_cycle(self, tmp_path: Path) -> None:
        xml = _xml(ChannelSpec("CD3", 4, "PE", clone="REA1"), version=0).replace('<M K="Cycle">4</M>', "")

        source = read_channel(_write_xml(tmp_path, xml))

        assert source is not None
        assert source.metadata.cycle == 4

    def test_v0_dye_reagent_has_no_clone(self, tmp_path: Path) -> None:
        assert _read(tmp_path, V0_CHANNELS[1], 0).clone is None

    def test_v1_falls_back_to_the_dye_for_name_and_fluorophore(self, tmp_path: Path) -> None:
        xml = (
            _xml(ChannelSpec("CD3", 1, "APC", clone="REA1151"))
            .replace('<M K="Biomarker">CD3</M>', "")
            .replace('<M K="Fluorochrome">APC</M>', "")
        )

        source = read_channel(_write_xml(tmp_path, xml))

        assert source is not None
        assert (source.metadata.name, source.metadata.fluorophore) == ("CD3", "CD3")

    def test_absent_optional_fields_are_none(self, tmp_path: Path) -> None:
        xml = (
            _xml(ChannelSpec("CD3", 1, "APC"))
            .replace('<M K="RoiId">001</M>', "")
            .replace('<M K="ExposureTime">100.0</M>', "")
            .replace('<M K="Fluorochrome">APC</M>', "")
            .replace('<M K="Dye">CD3</M>', "")
            .replace(
                '<Plate ID="Plate:0" Name="01"><Well ID="Well:0" Row="4" Column="1" ExternalIdentifier="D01"/>',
                '<Plate ID="Plate:0">',
            )
        )

        source = read_channel(_write_xml(tmp_path, xml))

        assert source is not None
        metadata = source.metadata
        assert (metadata.roi, metadata.exposure_ms, metadata.fluorophore, metadata.well) == (None, None, None, None)

    def test_pixel_positions_place_the_plane(self, tmp_path: Path) -> None:
        write_channel(tmp_path, ChannelSpec("CD3", 1, "APC", offset=(3, 5.0)), version=1, index=0)

        (path,) = tmp_path.iterdir()
        source = read_channel(ArtifactFile(path.resolve()))

        assert source is not None
        assert source.offset_yx == (3, 5)

    def test_nanometer_pixel_size(self, tmp_path: Path) -> None:
        source = read_channel(_write_xml(tmp_path, _xml(V1_CHANNELS[0], physical_size=("170", "nm"))))

        assert source is not None
        assert source.pixel_size_um == 0.17

    def test_tiff_without_ome_is_not_a_channel(self, tmp_path: Path) -> None:
        path = tmp_path / "preview.tif"
        tifffile.imwrite(path, np.zeros((3, 4), dtype=np.uint16), metadata=None)

        assert read_channel(ArtifactFile(path.resolve())) is None

    def test_benign_ome_id_warnings_are_filtered_locally(self, tmp_path: Path) -> None:
        path = _write_xml(tmp_path, _xml(V1_CHANNELS[0]))
        filters = list(warnings.filters)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            read_channel(path)

        assert not [warning for warning in caught if "ID" in str(warning.message)]
        assert warnings.filters == filters

    def test_inspects_each_file_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        inspect = mocker.spy(_metadata, "inspect_tiff")

        read_channel(_write_xml(tmp_path, _xml(V1_CHANNELS[0])))

        inspect.assert_called_once()

    def test_malformed_tiff_keeps_its_error_with_reader_context(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.tif"
        path.write_bytes(b"not a tiff")

        with pytest.raises(RasterFormatError) as caught:
            read_channel(ArtifactFile(path.resolve()))
        assert any("reader 'macsima'" in note for note in caught.value.__notes__)


_V1 = _xml(ChannelSpec("CD3", 1, "APC", clone="REA1151"))


class TestReadChannelRejects:
    """Malformed MACSima content fails with reader, version, and path context."""

    @pytest.mark.parametrize(
        ("xml", "match"),
        [
            pytest.param(
                _V1.replace('<M K="SoftwareVersion">1.1.1</M>', ""), r"software versions \[\]", id="no-version"
            ),
            pytest.param(_V1.replace(">1.1.1<", ">2.0.0<"), "major version 0 or 1", id="unsupported-major"),
            pytest.param(_V1.replace(">1.1.1<", ">one<"), "major version 0 or 1", id="non-numeric-version"),
            pytest.param(
                _V1.replace(
                    '<M K="SoftwareVersion">1.1.1</M>',
                    '<M K="SoftwareVersion">1.1.1</M><M K="Software version">v0.15.3</M>',
                ),
                r"software versions \['0\.15\.3', '1\.1\.1'\]",
                id="conflicting-version-keys",
            ),
            pytest.param(_V1.replace('<M K="ScanType">S</M>', ""), "no 'ScanType' annotation", id="no-scan-type"),
            pytest.param(
                _V1.replace('<M K="ScanType">S</M>', '<M K="ScanType">X</M>'), "image type 'X'", id="unknown-type"
            ),
            pytest.param(_V1.replace('<M K="Cycle">1</M>', ""), "no cycle annotation", id="no-cycle"),
            pytest.param(
                _V1.replace('<M K="Cycle">1</M>', '<M K="Cycle">1.5</M>'), "cycle '1.5'", id="fractional-cycle"
            ),
            pytest.param(
                _V1.replace('<M K="Cycle">1</M>', '<M K="Cycle">2</M>', 1),
                "conflicting values for 'Cycle'",
                id="conflicting-cycle",
            ),
            pytest.param(_V1.replace('<M K="RoiId">001</M>', '<M K="RoiId">-1</M>'), "ROI ID '-1'", id="negative-roi"),
            pytest.param(
                _V1.replace('<M K="Biomarker">CD3</M>', "").replace('<M K="Dye">CD3</M>', ""),
                "no marker name",
                id="no-name",
            ),
            pytest.param(
                _V1.replace('<M K="ExposureTimeUnit">ms</M>', '<M K="ExposureTimeUnit">s</M>'),
                "exposure unit 's'",
                id="exposure-unit",
            ),
            pytest.param(_V1.replace(">100.0<", ">fast<"), "exposure 'fast'", id="exposure-text"),
            pytest.param(_V1.replace(">100.0<", ">-1<"), "exposure '-1'", id="exposure-negative"),
            pytest.param(_V1.replace(">100.0<", ">inf<"), "exposure 'inf'", id="exposure-infinite"),
            pytest.param(_V1.replace('PhysicalSizeXUnit="µm"', 'PhysicalSizeXUnit="mm"'), "µm or nm", id="size-unit"),
            pytest.param(_V1.replace('PhysicalSizeY="0.17"', 'PhysicalSizeY="0.2"'), "square pixels", id="anisotropic"),
            pytest.param(_V1.replace(' PhysicalSizeX="0.17"', ""), "finite positive size", id="no-size"),
            pytest.param(
                _V1.replace(
                    '<Plane TheC="0" TheT="0" TheZ="0"/>',
                    '<Plane TheC="0" TheT="0" TheZ="0" PositionX="4" PositionY="2"/>',
                ),
                "pixel units",
                id="reference-frame-position",
            ),
            pytest.param(
                _V1.replace(
                    '<Plane TheC="0" TheT="0" TheZ="0"/>',
                    '<Plane TheC="0" TheT="0" TheZ="0" PositionX="4.5" PositionXUnit="pixel" PositionY="2" '
                    'PositionYUnit="pixel"/>',
                ),
                "integral PositionY and PositionX",
                id="fractional-position",
            ),
            pytest.param(
                _V1.replace(
                    '<Plane TheC="0" TheT="0" TheZ="0"/>',
                    '<Plane TheC="0" TheT="0" TheZ="0" PositionX="4" PositionXUnit="pixel"/>',
                ),
                "integral PositionY and PositionX",
                id="partial-position",
            ),
            pytest.param(
                _V1.replace('SizeX="15"', 'SizeX="16"'), "matching the OME SizeY and SizeX", id="size-mismatch"
            ),
        ],
    )
    def test_v1_contract(self, tmp_path: Path, xml: str, match: str) -> None:
        with pytest.raises(ReaderFormatError, match=match) as caught:
            read_channel(_write_xml(tmp_path, xml))
        assert "channel.tif" in str(caught.value)

    def test_errors_name_the_detected_version(self, tmp_path: Path) -> None:
        with pytest.raises(ReaderFormatError, match=r"detected version '1\.1\.1'"):
            read_channel(_write_xml(tmp_path, _V1.replace('<M K="ScanType">S</M>', "")))

    def test_version_keys_may_repeat_one_version_with_a_prefix(self, tmp_path: Path) -> None:
        xml = _V1.replace(
            '<M K="SoftwareVersion">1.1.1</M>', '<M K="SoftwareVersion">1.1.1</M><M K="Software version">v1.1.1</M>'
        )

        source = read_channel(_write_xml(tmp_path, xml))

        assert source is not None
        assert source.software_version == "1.1.1"

    def test_more_than_one_well(self, tmp_path: Path) -> None:
        well = '<Well ID="Well:0" Row="4" Column="1" ExternalIdentifier="D01"/>'
        xml = _V1.replace(well, well + well.replace("Well:0", "Well:1").replace("D01", "D02"))

        with pytest.raises(ReaderFormatError, match="2 plate wells"):
            read_channel(_write_xml(tmp_path, xml))

    def test_v0_rejects_more_than_one_reagent(self, tmp_path: Path) -> None:
        reagent = '<Reagent ID="Reagent:0" Name="CD15__VIMC6"/>'
        xml = _xml(V0_CHANNELS[0], version=0).replace(reagent, reagent + reagent.replace("Reagent:0", "Reagent:1"))

        with pytest.raises(ReaderFormatError, match="2 screen reagents"):
            read_channel(_write_xml(tmp_path, xml))

    def test_v0_requires_a_reagent_marker(self, tmp_path: Path) -> None:
        xml = _xml(V0_CHANNELS[0], version=0).replace('Name="CD15__VIMC6"', 'Name="__VIMC6"')

        with pytest.raises(ReaderFormatError, match="no marker name"):
            read_channel(_write_xml(tmp_path, xml))

    def test_more_than_one_plane(self, tmp_path: Path) -> None:
        xml = _V1.replace('SizeT="1"', 'SizeT="2"').replace('PlaneCount="1"', 'PlaneCount="2"')

        with pytest.raises(ReaderFormatError, match="one plane per channel file"):
            read_channel(_write_xml(tmp_path, xml, shape=(2, 12, 15)))

    def test_duplicate_plane_entries(self, tmp_path: Path) -> None:
        plane = '<Plane TheC="0" TheT="0" TheZ="0"/>'
        xml = _V1.replace(plane, plane * 2)

        with pytest.raises(ReaderFormatError, match="2 OME planes"):
            read_channel(_write_xml(tmp_path, xml))

    def test_more_than_one_image(self, tmp_path: Path) -> None:
        image_start = _V1.index("<Image ")
        image_end = _V1.index("</Image>") + len("</Image>")
        second = _V1[image_start:image_end].replace('IFD="0"', 'IFD="1"').replace("Image:0", "Image:1")
        second = second.replace("Pixels:0", "Pixels:1").replace("Channel:0:0", "Channel:1:0")
        xml = _V1[:image_end] + second + _V1[image_end:]
        path = tmp_path / "channel.tif"
        with tifffile.TiffWriter(path) as writer:
            for page in range(2):
                writer.write(
                    np.zeros((12, 15), dtype=np.uint16),
                    description=xml.encode() if page == 0 else None,
                    metadata=None,
                    photometric="minisblack",
                )

        with pytest.raises(ReaderFormatError, match="2 OME images"):
            read_channel(ArtifactFile(path.resolve()))
