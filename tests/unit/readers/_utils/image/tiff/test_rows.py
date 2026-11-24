"""Tests for direct whole-row access in TIFF storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import tifffile

from spatialdata_io.readers._utils.image.tiff._rows import direct_row_axis, read_contiguous_rows

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadContiguousRows:
    def test_imagej_planes_separated_by_ifds_use_individual_offsets(self, tmp_path: Path) -> None:
        path = tmp_path / "separate-planes.tif"
        planes = np.arange(2 * 9 * 5, dtype=np.uint16).reshape(2, 9, 5)
        description = "ImageJ=1.53c\nimages=2\nchannels=2\nhyperstack=true\nmode=grayscale\n"
        with tifffile.TiffWriter(path) as writer:
            writer.write(planes[0], description=description, metadata=None, rowsperstrip=9)
            writer.write(planes[1], metadata=None, rowsperstrip=9)

        with tifffile.TiffFile(path) as tiff:
            assert tiff.is_imagej
            assert tiff.pages[1].offset < tiff.pages.first.dataoffsets[0] + planes.nbytes
            series = tiff.series[0]
            assert direct_row_axis(tiff, series, planes.shape) == 1
            crop = (slice(1, 2), slice(4, 6), slice(0, 5))
            result = read_contiguous_rows(tiff, series, planes.shape, crop)

        np.testing.assert_array_equal(result, planes[crop])

    def test_replaced_page_uses_decoder_for_new_encoding(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "changed.tif"
        plane = np.arange(9 * 5, dtype=np.uint16).reshape(9, 5)
        tifffile.imwrite(path, plane, compression="deflate")

        with tifffile.TiffFile(path) as tiff:
            read_array = mocker.spy(tifffile.FileHandle, "read_array")
            crop = (slice(4, 6), slice(0, 5))
            assert read_contiguous_rows(tiff, tiff.pages.first, plane.shape, crop) is None
            read_array.assert_not_called()
