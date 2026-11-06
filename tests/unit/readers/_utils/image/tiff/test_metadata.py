"""Tests for TIFF inspection boundary validation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from ome_types import OME
from ome_types.model import Image, Pixels

from spatialdata_io.readers._utils.errors import RasterFormatError, ReaderErrorContext
from spatialdata_io.readers._utils.image import inspect_tiff
from spatialdata_io.readers._utils.image.tiff._metadata import _level_metadata, _sample_color
from spatialdata_io.readers._utils.paths import ArtifactFile

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

    from spatialdata_io.readers._utils.image._types import SampleColor


_RASTER_CONTEXT = ReaderErrorContext("test", "raster")


class TestInspectTiff:
    """Reject invalid sources before opening tifffile metadata structures."""

    @pytest.mark.parametrize("signature", [b"", b"II", b"not tiff"])
    def test_rejects_invalid_magic_bytes(self, tmp_path: Path, signature: bytes) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(signature)

        with pytest.raises(RasterFormatError, match="signature") as caught:
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_RASTER_CONTEXT.describe(path=path)}."]


class TestLevelMetadata:
    """Keep semantic dimensions stable when tifffile retains source singleton axes."""

    def test_unsqueezed_series_matches_squeezed_store(self, mocker: MockerFixture) -> None:
        level = mocker.Mock(spec=tifffile.TiffPageSeries)
        level.shape = (1, 1, 8, 1)
        level.axes = "TCYX"
        level.dtype = np.dtype(np.uint16)
        tiff = mocker.Mock(spec=tifffile.TiffFile)
        mocker.patch("spatialdata_io.readers._utils.image.tiff._metadata._native_chunks", return_value=(8, 1))
        metadata = _level_metadata(tiff, level, series_index=0, level_index=0)
        assert metadata.shape == (8, 1)
        assert metadata.axes == ("Y", "X")
        assert metadata.native_chunks == (8, 1)

    @pytest.mark.parametrize(
        ("shape", "chunks", "match"),
        [((0, 8), (1, 8), "invalid shape"), ((), (), "invalid shape"), ((4, 8), (4,), "invalid native chunks")],
    )
    def test_rejects_inconsistent_storage_facts(
        self, mocker: MockerFixture, shape: tuple[int, ...], chunks: tuple[int, ...], match: str
    ) -> None:
        level = mocker.Mock(spec=tifffile.TiffPageSeries)
        level.shape = shape
        level.axes = "YX"[: len(shape)]
        level.dtype = np.dtype(np.uint8)
        mocker.patch("spatialdata_io.readers._utils.image.tiff._metadata._native_chunks", return_value=chunks)

        with pytest.raises(RasterFormatError, match=match):
            _level_metadata(mocker.Mock(spec=tifffile.TiffFile), level, series_index=0, level_index=0)


class TestSampleColor:
    """Record only the colour model a level's first page declares for its samples axis."""

    @pytest.mark.parametrize(
        ("samples", "color"),
        [
            pytest.param((3, "rgb", None, None), "rgb", id="rgb"),
            pytest.param((4, "rgb", ["unassalpha"], None), "rgba", id="rgb-unassalpha"),
            pytest.param((4, "rgb", ["assocalpha"], None), "rgba", id="rgb-assocalpha"),
            pytest.param((4, "rgb", ["unspecified"], None), None, id="rgb-unspecified"),
            pytest.param((5, "rgb", ["unassalpha", "unassalpha"], None), None, id="rgb-two-extra"),
            pytest.param((3, "minisblack", None, None), None, id="minisblack"),
            pytest.param((3, "ycbcr", None, None), None, id="ycbcr-uncompressed"),
            # tifffile stores JPEG-compressed RGB as YCbCr, which decodes to RGB.
            pytest.param((3, "rgb", None, "jpeg"), "rgb", id="jpeg-ycbcr-from-rgb"),
            pytest.param((3, "ycbcr", None, "jpeg"), "rgb", id="jpeg-ycbcr"),
            pytest.param((1, "minisblack", None, None), None, id="grayscale"),
        ],
    )
    def test_records_declared_interleaved_samples(
        self,
        tmp_path: Path,
        samples: tuple[int, str, list[str] | None, str | None],
        color: SampleColor | None,
    ) -> None:
        count, photometric, extrasamples, compression = samples
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path,
            np.zeros((16, 16) if count == 1 else (16, 16, count), dtype=np.uint8),
            photometric=photometric,
            planarconfig="contig",
            extrasamples=extrasamples,
            compression=compression,
        )

        level = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT).series[0].levels[0]

        assert level.axes == (("Y", "X") if count == 1 else ("Y", "X", "S"))
        assert level.color == color

    def test_records_rgb_planes_and_every_pyramid_level(self, tmp_path: Path) -> None:
        path = tmp_path / "planar.tif"
        with tifffile.TiffWriter(path) as writer:
            planes = np.zeros((3, 32, 32), dtype=np.uint8)
            writer.write(planes, photometric="rgb", planarconfig="separate", tile=(16, 16), subifds=1)
            writer.write(planes[:, ::2, ::2], photometric="rgb", planarconfig="separate", tile=(16, 16), subfiletype=1)

        levels = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT).series[0].levels

        assert [(level.axes, level.color) for level in levels] == [(("S", "Y", "X"), "rgb")] * 2

    def test_planar_jpeg_ycbcr_declares_no_rgb(self, tmp_path: Path) -> None:
        # Separate YCbCr planes decode to luma and chroma, not RGB.
        path = tmp_path / "planar-ycbcr.tif"
        tifffile.imwrite(
            path,
            np.zeros((3, 16, 16), dtype=np.uint8),
            photometric="ycbcr",
            planarconfig="separate",
            compression="jpeg",
        )

        level = inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT).series[0].levels[0]

        assert level.axes == ("S", "Y", "X")
        assert level.color is None

    def test_rgb_without_extra_samples_names_only_three_samples(self, mocker: MockerFixture) -> None:
        # A malformed RGB page with a fourth sample but no ExtraSamples declares nothing about it.
        level = mocker.Mock(spec=tifffile.TiffPageSeries)
        level.keyframe = mocker.Mock(photometric=tifffile.PHOTOMETRIC.RGB, extrasamples=())

        assert _sample_color(level, ("Y", "X", "S"), (2, 2, 4)) is None


class TestParseOmeXml:
    def test_malformed_ome_xml_is_a_raster_format_error_with_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.ome.tif"
        # Mismatched tags: tifffile recognizes OME-XML, ome-types cannot parse it.
        description = (
            '<?xml version="1.0"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image></OME>'
        )
        tifffile.imwrite(path, np.zeros((2, 3), dtype=np.uint8), description=description, metadata=None)

        with pytest.raises(RasterFormatError, match="Cannot parse OME metadata") as caught:
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)

        assert caught.value.__notes__ == [f"While handling {_RASTER_CONTEXT.describe(path=path)}."]


class TestValidateOmeImages:
    def test_rejects_ome_image_count_differing_from_series(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.zeros((2, 3), dtype=np.uint8))
        pixels = Pixels(dimension_order="XYZCT", type="uint8", size_x=3, size_y=2, size_z=1, size_c=1, size_t=1)
        ome = OME(images=[Image(pixels=pixels), Image(pixels=pixels)])
        mocker.patch("spatialdata_io.readers._utils.image.tiff._metadata._parse_ome_xml", return_value=ome)

        with pytest.raises(RasterFormatError, match="declares 2 images, but tifffile discovered 1 image series"):
            inspect_tiff(ArtifactFile(path), context=_RASTER_CONTEXT)
