"""Native raster axes, lazy partial reads and serializable task-owned sources."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from xarray import DataArray

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq._images import read_image
from spatialdata_io.readers.stereoseq._layout import GridImage

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestReadImage:
    """Preserve source pixels while keeping construction free of pixel tasks."""

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_native_bounded_lazy_reads(self, tmp_path: Path, mocker: MockerFixture, compression: str | None) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(128 * 192, dtype=np.uint16).reshape(128, 192)
        tifffile.imwrite(path, pixels, tile=(32, 32), compression=compression, metadata={"axes": "YX"})

        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            image = read_image(GridImage(ArtifactFile(path), "registered"), chunks=None)
        assert isinstance(image, DataArray)
        tasks.assert_not_called()
        reads.assert_not_called()
        assert image.dims == ("c", "y", "x")
        assert image.data.chunksize == (1, 32, 32)
        assert image.dtype == np.uint16
        restored = pickle.loads(pickle.dumps(image.data))
        assert_array_equal(restored[:, 2:5, 3:7].compute(scheduler="synchronous"), pixels[None, 2:5, 3:7])
        reads.assert_called_once()
        assert reads.call_args.args[1] == (slice(0, 32), slice(0, 32))

    @pytest.mark.parametrize(("axes", "shape"), [("YXS", (16, 24, 3)), ("CYX", (7, 16, 24)), ("TCYX", (1, 2, 16, 24))])
    def test_channels_and_singletons(self, tmp_path: Path, axes: str, shape: tuple[int, ...]) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(path, pixels, metadata={"axes": axes}, photometric="rgb" if axes == "YXS" else "minisblack")

        image = read_image(GridImage(ArtifactFile(path), "registered"), chunks=(1, 8, 8))
        assert isinstance(image, DataArray)
        expected = pixels.transpose(2, 0, 1) if axes == "YXS" else pixels[0] if axes == "TCYX" else pixels
        assert_array_equal(image.data.compute(), expected)
        assert image.data.chunksize == (1, 8, 8)
        if axes == "YXS":
            assert image.c.values.tolist() == ["r", "g", "b"]

    def test_rejects_multiple_series(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.zeros((6, 7), dtype=np.uint8), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="series"):
            read_image(GridImage(ArtifactFile(path), "registered"), chunks=None)

    def test_rejects_unknown_nonsingleton_axis(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        tifffile.imwrite(
            path, np.zeros((5, 16, 24), dtype=np.uint8), metadata={"axes": "QYX"}, photometric="minisblack"
        )

        with pytest.raises(ValueError, match=r"[Qq]"):
            read_image(GridImage(ArtifactFile(path), "registered"), chunks=None)
