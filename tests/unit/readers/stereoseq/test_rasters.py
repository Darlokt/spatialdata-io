"""Image and mask-label axes, chunks, pyramids, lazy partial reads and task-owned sources."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from numpy.testing import assert_array_equal
from spatialdata.transformations import Identity, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.stereoseq._layout import DistanceMask, GridImage
from spatialdata_io.readers.stereoseq._rasters import read_image, read_mask

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

DEFAULT_OPTIONS = make_raster_options()


class TestReadImage:
    """Preserve source pixels while keeping construction free of pixel tasks."""

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_bounded_lazy_reads(self, tmp_path: Path, mocker: MockerFixture, compression: str | None) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(128 * 192, dtype=np.uint16).reshape(128, 192)
        tifffile.imwrite(path, pixels, tile=(32, 32), compression=compression, metadata={"axes": "YX"})

        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        # Chunks equal to the stored tiles; default tiles would merge them all.
        with Callback(posttask=tasks):
            image = read_image(
                GridImage(ArtifactFile(path), "registered"), raster=make_raster_options(chunks=(1, 32, 32))
            )
        assert isinstance(image, DataArray)
        tasks.assert_not_called()
        reads.assert_not_called()
        assert image.dims == ("c", "y", "x")
        assert image.data.chunksize == (1, 32, 32)
        assert image.dtype == np.uint16
        restored = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Only deserialize this test's own graph.
        assert_array_equal(restored[:, 2:5, 3:7].compute(scheduler="synchronous"), pixels[None, 2:5, 3:7])
        reads.assert_called_once()
        assert reads.call_args.args[1] == (slice(0, 32), slice(0, 32))

    @pytest.mark.parametrize(("axes", "shape"), [("YXS", (16, 24, 3)), ("CYX", (7, 16, 24)), ("TCYX", (1, 2, 16, 24))])
    def test_channels_and_singletons(self, tmp_path: Path, axes: str, shape: tuple[int, ...]) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(path, pixels, metadata={"axes": axes}, photometric="rgb" if axes == "YXS" else "minisblack")

        image = read_image(GridImage(ArtifactFile(path), "registered"), raster=make_raster_options(chunks=(1, 8, 8)))
        assert isinstance(image, DataArray)
        expected = pixels.transpose(2, 0, 1) if axes == "YXS" else pixels[0] if axes == "TCYX" else pixels
        assert_array_equal(image.data.compute(), expected)
        assert image.data.chunksize == (1, 8, 8)
        if axes == "YXS":
            assert image.c.values.tolist() == ["r", "g", "b"]

    def test_default_options_are_single_scale_channel_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
        tifffile.imwrite(path, pixels, photometric="rgb", tile=(16, 16))

        image = read_image(GridImage(ArtifactFile(path), "registered"), raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert image.data.chunks == ((1, 1, 1), (16,), (24,))
        assert_array_equal(image.data.compute(), pixels.transpose(2, 0, 1))

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        pixels = np.arange(16 * 24, dtype=np.uint16).reshape(16, 24)
        tifffile.imwrite(path, pixels, metadata={"axes": "YX"})

        image = read_image(GridImage(ArtifactFile(path), "registered"), raster=make_raster_options((2, 2), (1, 8, 8)))

        assert isinstance(image, DataTree)
        assert list(image.children) == ["scale0", "scale1", "scale2"]
        scale0 = image["scale0"]["image"]
        assert scale0.data.chunksize == (1, 8, 8)
        assert_array_equal(scale0.data.compute(), pixels[None])
        assert image["scale1"]["image"].shape == (1, 8, 12)
        assert image["scale2"]["image"].shape == (1, 4, 6)

    def test_rejects_multiple_series(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.zeros((4, 5), dtype=np.uint8), metadata={"axes": "YX"})
            writer.write(np.zeros((6, 7), dtype=np.uint8), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="series"):
            read_image(GridImage(ArtifactFile(path), "registered"), raster=DEFAULT_OPTIONS)

    def test_rejects_unknown_nonsingleton_axis(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_regist.tif"
        tifffile.imwrite(
            path, np.zeros((5, 16, 24), dtype=np.uint8), metadata={"axes": "QYX"}, photometric="minisblack"
        )

        with pytest.raises(ValueError, match=r"[Qq]"):
            read_image(GridImage(ArtifactFile(path), "registered"), raster=DEFAULT_OPTIONS)


class TestReadMask:
    """Return masks as lazy YX labels whose every pyramid level keeps source values."""

    def test_lazy_exact_labels(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "sample_mask.tif"
        pixels = (np.arange(64 * 96).reshape(64, 96) % 3 == 0).astype(np.uint8)
        tifffile.imwrite(path, pixels, tile=(32, 32), compression="deflate", metadata={"axes": "YX"})

        reads = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            labels = read_mask(GridImage(ArtifactFile(path), "mask"), raster=make_raster_options(chunks=(1, 32, 32)))
        assert isinstance(labels, DataArray)
        tasks.assert_not_called()
        reads.assert_not_called()
        assert labels.dims == ("y", "x")
        assert labels.dtype == np.uint8
        assert labels.data.chunksize == (32, 32)
        assert isinstance(get_transformation(labels), Identity)
        assert_array_equal(labels.data[2:5, 3:7].compute(scheduler="synchronous"), pixels[2:5, 3:7])
        reads.assert_called_once()

    def test_pyramid_keeps_source_values(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_tissue_cut.tif"
        pixels = np.zeros((16, 24), dtype=np.uint8)
        pixels[::2, ::2] = 255
        tifffile.imwrite(path, pixels, metadata={"axes": "YX"})

        labels = read_mask(GridImage(ArtifactFile(path), "tissue_cut"), raster=make_raster_options((2, 2), (1, 8, 8)))

        assert isinstance(labels, DataTree)
        assert list(labels.children) == ["scale0", "scale1", "scale2"]
        assert labels["scale0"]["image"].data.chunksize == (8, 8)
        assert_array_equal(labels["scale0"]["image"].data.compute(), pixels)
        assert labels["scale2"]["image"].shape == (4, 6)
        for level in ("scale1", "scale2"):
            values = labels[level]["image"].data.compute()
            assert values.dtype == np.uint8
            # Interpolated downsampling would produce values strictly between 0 and 255.
            assert set(np.unique(values).tolist()) <= {0, 255}

    @pytest.mark.parametrize("dtype", [np.bool_, np.int16, np.uint32])
    def test_boolean_and_integer_dtypes(self, tmp_path: Path, dtype: type[np.generic]) -> None:
        path = tmp_path / "HE_sample_mask_edm_dis_10.tif"
        pixels = (np.arange(16 * 24).reshape(16, 24) % 5).astype(dtype)
        tifffile.imwrite(path, pixels, metadata={"axes": "YX"})

        labels = read_mask(DistanceMask(ArtifactFile(path), "HE", 10), raster=make_raster_options((2,), 8))

        assert isinstance(labels, DataTree)
        scale0 = labels["scale0"]["image"]
        assert scale0.dtype == dtype
        assert_array_equal(scale0.data.compute(), pixels)

    def test_squeezes_singleton_channel(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_mask.tif"
        pixels = np.arange(16 * 24, dtype=np.uint16).reshape(1, 16, 24)
        tifffile.imwrite(path, pixels, metadata={"axes": "CYX"}, photometric="minisblack")

        labels = read_mask(GridImage(ArtifactFile(path), "mask"), raster=DEFAULT_OPTIONS)

        assert isinstance(labels, DataArray)
        assert labels.dims == ("y", "x")
        assert_array_equal(labels.data.compute(), pixels[0])

    def test_rejects_multichannel(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_mask.tif"
        tifffile.imwrite(path, np.zeros((16, 24, 3), dtype=np.uint8), photometric="rgb")

        with pytest.raises(ValueError, match=r"[Ss]"):
            read_mask(GridImage(ArtifactFile(path), "mask"), raster=DEFAULT_OPTIONS)

    def test_rejects_floating_point(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_mask.tif"
        tifffile.imwrite(path, np.zeros((16, 24), dtype=np.float32), metadata={"axes": "YX"})

        with pytest.raises(ReaderFormatError, match="float32"):
            read_mask(GridImage(ArtifactFile(path), "mask"), raster=DEFAULT_OPTIONS)

    def test_64_bit_values_require_single_scale(self, tmp_path: Path) -> None:
        path = tmp_path / "sample_mask.tif"
        pixels = np.full((16, 24), 2**60 + 1, dtype=np.uint64)
        tifffile.imwrite(path, pixels, metadata={"axes": "YX"})
        source = GridImage(ArtifactFile(path), "mask")

        with pytest.raises(ValueError, match="32-bit"):
            read_mask(source, raster=make_raster_options((2,), 8))
        labels = read_mask(source, raster=DEFAULT_OPTIONS)
        assert isinstance(labels, DataArray)
        assert_array_equal(labels.data.compute(), pixels)
