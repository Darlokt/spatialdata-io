"""Validate QPI IFD geometry and quantitative channel identity."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from spatialdata.transformations import Identity
from xarray import DataArray

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import require_file
from spatialdata_io.readers.phenocycler._images import inspect_scan, read_scan, scan_attributes
from tests.integration.readers.phenocycler._fixtures import QpiPage, write_pages, write_qptiff

if TYPE_CHECKING:
    from pathlib import Path

_CONTEXT = ReaderErrorContext("phenocycler", "QPI scan")
_PIXELS = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
_BASE = QpiPage(_PIXELS, name="Fluor", marker="CD3", stage_position=(1.0, 2.0))


class TestInspectScan:
    @pytest.mark.parametrize(
        ("pages", "message"),
        [
            ((replace(_BASE, image_type="Thumbnail"),), "no FullResolution pages"),
            ((_BASE, replace(_BASE, pixels=_PIXELS[:, :12])), "mismatched shape or dtype"),
            ((_BASE, replace(_BASE, pixels=_PIXELS.astype("uint8"))), "mismatched shape or dtype"),
            ((_BASE, replace(_BASE, resolution=(10000, 10000))), "conflicting pixel sizes"),
            ((_BASE, replace(_BASE, stage_position=(1.0, 1.9))), "conflicting stage positions"),
            ((replace(_BASE, resolution=(20000, 40000)),), "matching X and Y pixel sizes"),
            ((_BASE, replace(_BASE, pixels=np.zeros((32, 48, 3), dtype=np.uint8))), "one top-left RGB"),
            ((replace(_BASE, description="<wrong>bad</wrong>"),), "unrecognized QPI ImageDescription"),
        ],
    )
    def test_bad_metadata_raises(self, tmp_path: Path, pages: tuple[QpiPage, ...], message: str) -> None:
        path = tmp_path / "bad.qptiff"
        write_pages(path, pages)
        with pytest.raises(ReaderFormatError, match=message) as exc:
            inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        assert exc.value.path == path

    def test_rejects_a_tiff_without_qpi_signature(self, tmp_path: Path) -> None:
        path = tmp_path / "ordinary.tif"
        tifffile.imwrite(path, _PIXELS)
        with pytest.raises(ReaderFormatError, match="QPI Software signature"):
            inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)

    def test_rejects_qpi_inverted_monochrome(self, tmp_path: Path) -> None:
        path = tmp_path / "inverted.tif"
        tifffile.imwrite(
            path,
            _PIXELS,
            photometric="miniswhite",
            software="PerkinElmer-QPI",
            description=(
                "<PerkinElmer-QPI-ImageDescription><ImageType>FullResolution</ImageType>"
                "</PerkinElmer-QPI-ImageDescription>"
            ),
        )
        with pytest.raises(ReaderFormatError, match="photometric 0"):
            inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)

    def test_conflicting_slide_identifiers_report_detected_version(self, tmp_path: Path) -> None:
        path = tmp_path / "different-slides.qptiff"
        second = replace(
            _BASE,
            description=(
                "<PerkinElmer-QPI-ImageDescription><ImageType>FullResolution</ImageType>"
                "<SlideID>another-slide</SlideID></PerkinElmer-QPI-ImageDescription>"
            ),
        )
        write_pages(path, [_BASE, second])
        with pytest.raises(ReaderFormatError, match="conflicting SlideID") as exc:
            inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        assert exc.value.context.detected_version == "5"

    def test_duplicate_name_resolution_reserves_source_names(self, tmp_path: Path) -> None:
        path = tmp_path / "duplicates.qptiff"
        expected = write_qptiff(
            path,
            markers=("CD3", "CD3", "CD3 (2)", None, "F4/80"),
            names=("Atto", "Atto", "Atto", "/private/path", "Atto"),
        )
        scan = inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        assert scan.channel_ids == ("CD3", "CD3 (3)", "CD3 (2)", "channel_3", "F4/80")
        assert scan.channels[1].metadata["band_name"] == "Atto"
        assert "band_name" not in scan.channels[3].metadata
        data = read_scan(
            scan,
            options=make_raster_options(chunks=(1, 16, 16)),
            channel_names=None,
            transformations={"image_pixels": Identity()},
            context=_CONTEXT,
        )
        assert isinstance(data, DataArray)
        np.testing.assert_array_equal(data.data.compute(), expected)


class TestReadScan:
    def test_stacks_contiguous_full_resolution_ifds(self, tmp_path: Path) -> None:
        path = tmp_path / "channels.qptiff"
        write_pages(path, [_BASE, replace(_BASE, pixels=_PIXELS + 1, marker="CD4")])
        scan = inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        data = read_scan(
            scan,
            options=make_raster_options(),
            channel_names=None,
            transformations={"image_pixels": Identity()},
            context=_CONTEXT,
        )
        assert isinstance(data, DataArray)
        np.testing.assert_array_equal(data.data.compute(), np.stack((_PIXELS, _PIXELS + 1)))

    def test_ignores_same_shape_auxiliary_ifd(self, tmp_path: Path) -> None:
        path = tmp_path / "interleaved.qptiff"
        other = replace(_BASE, pixels=_PIXELS + 1, marker="CD4")
        thumbnail = replace(_BASE, pixels=_PIXELS + 2, image_type="Thumbnail")
        write_pages(path, [_BASE, thumbnail, other])
        scan = inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        assert scan.pages == (0, 2)
        data = read_scan(
            scan,
            options=make_raster_options(),
            channel_names=None,
            transformations={"image_pixels": Identity()},
            context=_CONTEXT,
        )
        assert isinstance(data, DataArray)
        np.testing.assert_array_equal(data.data.compute(), np.stack((_PIXELS, _PIXELS + 1)))


class TestScanAttributes:
    def test_rejects_override_with_incorrect_length(self, tmp_path: Path) -> None:
        path = tmp_path / "channels.qptiff"
        write_pages(path, [_BASE])
        scan = inspect_scan(require_file(path, context=_CONTEXT), context=_CONTEXT)
        with pytest.raises(ValueError, match="has 2 entries"):
            scan_attributes(scan, ("first", "second"))
