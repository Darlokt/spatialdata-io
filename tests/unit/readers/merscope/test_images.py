"""Lazy MERSCOPE mosaics, grayscale axes, native partial I/O and resource ownership."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import Identity, get_transformation
from xarray import DataArray, DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.merscope._images import _read_channel, make_raster_options, read_images
from spatialdata_io.readers.merscope._layout import MosaicChannel, MosaicLayer

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray
    from pytest_mock import MockerFixture


def _write_layer(root: Path, z: int, *, compression: str | None = None) -> tuple[MosaicLayer, NDArray[np.uint16]]:
    """Write distinguishable channels with four-by-three native tile grids."""
    source = np.arange(2 * 48 * 64, dtype=np.uint16).reshape(2, 48, 64) + z * 100
    channels = []
    for index, stain in enumerate(("Cyto", "DAPI")):
        path = root / f"mosaic_{stain}_z{z}.tif"
        tifffile.imwrite(path, source[index], tile=(16, 16), photometric="minisblack", compression=compression)
        channels.append(MosaicChannel(stain, ArtifactFile(path)))
    return MosaicLayer(z, tuple(channels)), source


class TestMakeRasterOptions:
    """Own ordered factors and reject invalid chunk shapes before metadata I/O."""

    def test_owned_sequences_and_valid_chunks(self) -> None:
        factors = [3, 2]
        options = make_raster_options(factors, (1, 16, 32))
        factors.reverse()
        assert options.scale_factors == (3, 2)
        assert options.chunks == (1, 16, 32)
        assert make_raster_options(None, None).scale_factors is None
        assert make_raster_options((2, 3), 16).chunks == 16

    @pytest.mark.parametrize("factors", ["2", pytest.param(b"2", id="bytes"), {2, 3}, {2: 3}, iter([2, 3])])
    def test_unordered_or_scalar_factors(
        self, factors: str | bytes | set[int] | dict[int, int] | Iterator[int]
    ) -> None:
        with pytest.raises(TypeError, match="ordered sequence"):
            make_raster_options(factors, None)  # type: ignore[bad-argument-type, arg-type]  # Exercise untyped public values.

    @pytest.mark.parametrize("factors", [[], [0], [1], [-1], [True], [2.0]])
    def test_invalid_factor_values(self, factors: list[float]) -> None:
        with pytest.raises(ValueError, match="scale_factors"):
            make_raster_options(factors, None)  # type: ignore[bad-argument-type, arg-type]  # Require integral factors greater than one.

    @pytest.mark.parametrize("chunks", [0, -1, (), (1, 2), (1, 0, 2), True, 2.0, "auto", (1, 2.0, 3)])
    def test_invalid_chunks(self, chunks: float | str | tuple[int | float, ...]) -> None:
        with pytest.raises((TypeError, ValueError), match="chunks"):
            make_raster_options(None, chunks)  # type: ignore[bad-argument-type, arg-type]  # Public validation precedes TIFF discovery.


class TestReadChannel:
    """Resolve full-resolution grayscale axis semantics from closed TIFF metadata."""

    def test_singleton_nonspatial_axes_and_spatial_singletons(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "single.tif")
        source = np.arange(16, dtype=np.uint16).reshape(1, 1, 16)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "CYX"})
        result = _read_channel(path)
        assert result.shape == (1, 16)
        np.testing.assert_array_equal(result.compute(), source[0])

    @pytest.mark.parametrize("invalid", ["axes", "dtype", "series", "encoding"])
    def test_invalid_source_metadata(self, tmp_path: Path, invalid: str) -> None:
        path = ArtifactFile(tmp_path / "channel.tif")
        if invalid == "encoding":
            path.write_bytes(b"not TIFF")
        elif invalid == "series":
            with tifffile.TiffWriter(path) as output:
                output.write(np.zeros((8, 8), dtype=np.uint16))
                output.write(np.zeros((4, 4), dtype=np.uint16))
        elif invalid == "axes":
            tifffile.imwrite(
                path, np.zeros((2, 8, 8), dtype=np.uint16), photometric="minisblack", metadata={"axes": "ZYX"}
            )
        else:
            tifffile.imwrite(path, np.zeros((8, 8), dtype=np.complex64), photometric="minisblack")
        with pytest.raises(ValueError, match=r"TIFF|merscope") as caught:
            _read_channel(path)
        assert "merscope" in str(caught.value) or any("merscope" in note for note in caught.value.__notes__)


class TestReadImages:
    """Keep all source graphs lazy and native tiles bounded through the image model."""

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_native_selection_channel_and_z_order_and_serialization(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        first, source = _write_layer(tmp_path, 2, compression=compression)
        second, _ = _write_layer(tmp_path, 0, compression=compression)
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()
        with Callback(posttask=tasks):
            result = read_images((first, second), dataset_id="slide_region", options=make_raster_options(None, None))
        tasks.assert_not_called()
        pixels.assert_not_called()
        assert list(result) == ["slide_region_z2", "slide_region_z0"]
        image = result["slide_region_z2"]
        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.dtype == np.dtype(np.uint16)
        assert image.data.chunks == ((1, 1), (16, 16, 16), (16, 16, 16, 16))
        assert image.coords["c"].values.tolist() == ["Cyto", "DAPI"]
        assert get_transformation(image, to_coordinate_system="global") == Identity()
        restored = pickle.loads(pickle.dumps(image.data))  # noqa: S301, RUF100 - Test-owned lazy graph only.
        np.testing.assert_array_equal(restored[:, :4, :6].compute(), source[:, :4, :6])
        assert pixels.call_count == 2

    def test_no_requested_layers_is_empty(self) -> None:
        assert read_images((), dataset_id="slide_region", options=make_raster_options(None, None)) == {}

    @pytest.mark.parametrize("invalid", ["shape", "dtype"])
    def test_selected_stains_and_planes_must_share_grid_and_dtype(self, tmp_path: Path, invalid: str) -> None:
        first, _ = _write_layer(tmp_path, 0)
        second, _ = _write_layer(tmp_path, 2)
        path = second.channels[-1].path
        tifffile.imwrite(
            path,
            np.zeros(
                (47, 64) if invalid == "shape" else (48, 64), dtype=np.uint32 if invalid == "dtype" else np.uint16
            ),
            photometric="minisblack",
        )
        with pytest.raises(ReaderFormatError, match="common shape"):
            read_images((first, second), dataset_id="region", options=make_raster_options(None, None))

    def test_pyramids_and_output_chunks_are_explicit(self, tmp_path: Path) -> None:
        layer, source = _write_layer(tmp_path, 0)
        plain = read_images((layer,), dataset_id="region", options=make_raster_options(None, (2, 12, 16)))["region_z0"]
        assert isinstance(plain, DataArray)
        assert plain.data.chunks == ((2,), (12, 12, 12, 12), (16, 16, 16, 16))
        np.testing.assert_array_equal(plain.data.compute(), source)
        pyramid = read_images((layer,), dataset_id="region", options=make_raster_options([2, 2], 16))["region_z0"]
        assert isinstance(pyramid, DataTree)
        assert list(pyramid.children) == ["scale0", "scale1", "scale2"]
        with pytest.raises(ValueError, match="spatial extent"):
            read_images((layer,), dataset_id="region", options=make_raster_options([64], None))

    @pytest.mark.parametrize("fail", [False, True])
    def test_task_resources_close_on_success_and_failure(
        self, tmp_path: Path, mocker: MockerFixture, *, fail: bool
    ) -> None:
        layer, _ = _write_layer(tmp_path, 0)
        image = read_images((layer,), dataset_id="region", options=make_raster_options(None, None))["region_z0"]
        assert isinstance(image, DataArray)
        with tifffile.TiffFile(layer.channels[0].path) as source:
            store = source.aszarr(series=0, level=0, squeeze=True)
            close = mocker.spy(store, "close")
            mocker.patch.object(source, "aszarr", return_value=store)
            mocker.patch.object(tiff_dask.tifffile, "TiffFile", return_value=source)
            if fail:
                mocker.patch.object(tiff_dask.zarr, "open_array", side_effect=RuntimeError("injected"))
                with pytest.raises(RuntimeError, match="injected"):
                    image.data[:1, :4, :6].compute(scheduler="synchronous")
            else:
                image.data[:1, :4, :6].compute(scheduler="synchronous")
            close.assert_called_once()
            assert source.filehandle.closed
