"""Lazy image placement, selection and bounded exact CytAssist sampling."""

from __future__ import annotations

import pickle
from dataclasses import replace
from typing import TYPE_CHECKING

import dask.array as da
import imagecodecs
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from scipy.ndimage import map_coordinates
from spatialdata.transformations import Affine, get_transformation
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import LazyRaster, inspect_tiff, make_raster_options, read_tiff
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._images import (
    ImageOptions,
    _normalize,
    _read_fullres,
    _warp_grid,
    make_image_options,
    read_images,
    warp_cytassist,
)
from spatialdata_io.readers.visium_hd._layout import ImageLayout, discover_layout
from spatialdata_io.readers.visium_hd._metadata import (
    ScaleFactors,
    SourceMetadata,
    read_scale_factors,
    read_source_metadata,
)
from spatialdata_io.readers.visium_hd._positions import fit_slide_frame
from spatialdata_io.readers.visium_hd._transforms import SlideFrame, apply_projective
from tests.integration.readers.visium_hd._fixtures import _write_png, write_bundle

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


def _options() -> ImageOptions:
    return make_image_options(scale_factors=(), chunks=(1, 16, 16), data_axes=None, tiff_series=None)


def _identity_frame() -> SlideFrame:
    return SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.03, 0.01)


def _bundle(root: Path) -> tuple[ImageLayout, SlideFrame, ScaleFactors, SourceMetadata]:
    write_bundle(root, images=True, fullres="root")
    layout = discover_layout(root, include_cytassist_image=True)
    assert layout.images is not None
    assert layout.feature_slice is not None
    return (
        layout.images,
        fit_slide_frame(layout.base_positions, (17, 17)),
        read_scale_factors(layout.base_scale_factors),
        read_source_metadata(layout.feature_slice),
    )


class TestMakeImageOptions:
    """Public selection is validated and owned before opening files."""

    def test_owned_axes_and_factors(self) -> None:
        axes, factors = ["C", "Y", "X"], [2]
        result = make_image_options(scale_factors=factors, chunks=8, data_axes=axes, tiff_series=0)
        axes[0], factors[0] = "S", 4
        assert result.data_axes == ("c", "y", "x")
        assert result.raster.scale_factors == (2,)

    @pytest.mark.parametrize("axes", [(), ("Y", "Y"), ("Y", "XX"), ("1", "Y")])
    def test_invalid_axes(self, axes: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            make_image_options(scale_factors=(), chunks=8, data_axes=axes, tiff_series=None)

    @pytest.mark.parametrize("axes", [17, [1], b"YX"])
    def test_invalid_axes_types(self, axes: object) -> None:
        with pytest.raises(TypeError, match="data_axes must be a sequence"):
            make_image_options(scale_factors=(), chunks=8, data_axes=axes, tiff_series=None)  # pyrefly: ignore[bad-argument-type] - Runtime option validation.

    @pytest.mark.parametrize("series", [True, -1, 1.5])
    def test_invalid_series(self, series: object) -> None:
        with pytest.raises(ValueError, match="tiff_series"):
            make_image_options(scale_factors=(), chunks=8, data_axes=None, tiff_series=series)  # pyrefly: ignore[bad-argument-type] - Runtime option validation.


class TestReadFullres:
    """TIFF family recognition uses content; encoded sources remain shared."""

    def test_tiff_content_and_series(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "unexpected.bin")
        tifffile.imwrite(path, np.ones((8, 9), np.uint16), bigtiff=True)
        raster = _read_fullres(path, _options())
        np.testing.assert_array_equal(raster.data.compute(), np.ones((8, 9), np.uint16))

    def test_ambiguous_series_requires_selection(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "multiple.tif")
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((8, 9), np.uint16))
            writer.write(np.ones((5, 6), np.uint16))
        with pytest.raises(ValueError, match="series"):
            _read_fullres(path, _options())
        raster = _read_fullres(path, replace(_options(), tiff_series=1))
        assert raster.data.shape == (5, 6)

    def test_png_and_jpeg(self, tmp_path: Path) -> None:
        png = ArtifactFile(tmp_path / "image.png")
        _write_png(png, np.zeros((8, 9, 3), np.uint8))
        assert _read_fullres(png, _options()).data.shape == (8, 9, 3)
        jpeg = ArtifactFile(tmp_path / "image.jpg")
        jpeg.write_bytes(imagecodecs.jpeg_encode(np.zeros((8, 9, 3), np.uint8)))
        assert _read_fullres(jpeg, _options()).data.shape == (8, 9, 3)
        with pytest.raises(ValueError, match="only to TIFF"):
            _read_fullres(png, replace(_options(), tiff_series=0))

    def test_unknown_content(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.xyz")
        path.write_bytes(b"unknown")
        with pytest.raises(ValueError, match="Unsupported"):
            _read_fullres(path, _options())


class TestNormalize:
    """Axes follow metadata and explicit names, including singleton removal."""

    def test_singleton_and_missing_channel(self) -> None:
        raster = LazyRaster(da.zeros((1, 5, 7)), ("t", "y", "x"))
        assert _normalize(raster).shape == (1, 5, 7)
        assert _normalize(LazyRaster(da.zeros((5, 7, 4)), ("y", "x", "s"))).shape == (4, 5, 7)

    def test_multiple_channels(self) -> None:
        raster = LazyRaster(da.zeros((2, 3, 5, 7)), ("c", "s", "y", "x"))
        with pytest.raises(ValueError, match="at most one"):
            _normalize(raster)

    def test_nonspatial_nonsingleton(self) -> None:
        raster = LazyRaster(da.zeros((2, 5, 7)), ("z", "y", "x"))
        with pytest.raises(ValueError, match="explicit selector"):
            _normalize(raster)


class TestReadImages:
    """Microscopy keeps source pixels and exact scales; CytAssist resamples lazily."""

    def test_pixels_dtypes_affines_and_zero_tasks(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        tasks: list[object] = []
        with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
            images = read_images(layout, frame, scales, source, _options())
        assert not tasks
        assert set(images) == {"hires_image", "lowres_image", "full_image", "cytassist_image"}
        for name, scale in (("hires_image", scales.hires), ("lowres_image", scales.lowres), ("full_image", 1.0)):
            image = images[name]
            assert isinstance(image, DataArray)
            transform = get_transformation(image, "global")
            assert isinstance(transform, Affine)
            np.testing.assert_array_equal(
                transform.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")),
                frame.image_pixels_to_global @ np.diag([1 / scale, 1 / scale, 1]),
            )
            assert image.attrs["registration_max_residual_um"] == frame.max_residual_um
        full = images["full_image"]
        assert isinstance(full, DataArray)
        assert full.dtype == np.uint16
        expected = np.arange(96 * 96 * 3, dtype=np.uint16).reshape(96, 96, 3).transpose(2, 0, 1)
        np.testing.assert_array_equal(full.data.compute(), expected)
        roundtrip = pickle.loads(pickle.dumps(full.data))  # noqa: S301, RUF100 - Test-owned lazy graph.
        np.testing.assert_array_equal(roundtrip[:, :2, :3].compute(), expected[:, :2, :3])

    def test_requested_pyramid_only_full_image(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        options = replace(_options(), raster=make_raster_options((2,), (1, 16, 16)))
        images = read_images(layout, frame, scales, source, options)
        assert isinstance(images["full_image"], DataTree)
        assert isinstance(images["hires_image"], DataArray)
        assert isinstance(images["cytassist_image"], DataArray)

    def test_missing_images_and_absent_fullres_options(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        empty = replace(layout, fullres=None, hires=None, lowres=None, cytassist=None)
        assert read_images(empty, frame, scales, source, _options()) == {}
        with pytest.raises(ValueError, match="require a full-resolution"):
            read_images(empty, frame, scales, source, replace(_options(), tiff_series=0))

    def test_longest_side_mismatch(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.fullres is not None
        tifffile.imwrite(layout.fullres, np.zeros((98, 90, 3), np.uint16), photometric="rgb")
        with pytest.raises(ReaderFormatError, match="longest sides"):
            read_images(layout, frame, scales, source, _options())

    def test_short_side_can_differ_and_fullres_without_hires(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.fullres is not None
        tifffile.imwrite(layout.fullres, np.zeros((96, 89, 3), np.uint16), photometric="rgb")
        images = read_images(layout, frame, scales, source, _options())
        assert images["full_image"].shape == (3, 96, 89)
        assert "full_image" in read_images(replace(layout, hires=None), frame, scales, source, _options())

    @pytest.mark.parametrize("missing_metadata", [False, True])
    def test_absent_cytassist_transform(self, tmp_path: Path, *, missing_metadata: bool) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        selected_source = None if missing_metadata else replace(source, cytassist=None)
        with pytest.raises(ReaderFormatError, match="null CytAssist"):
            read_images(layout, frame, scales, selected_source, _options())

    def test_ambiguous_cytassist_series(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.cytassist is not None
        with tifffile.TiffWriter(layout.cytassist) as writer:
            writer.write(np.ones((8, 9), np.uint8))
            writer.write(np.ones((5, 6), np.uint8))
        with pytest.raises(ReaderFormatError, match="unambiguous CytAssist"):
            read_images(layout, frame, scales, source, _options())

    def test_microscopy_without_feature_slice(self, tmp_path: Path) -> None:
        layout, frame, scales, _ = _bundle(tmp_path)
        images = read_images(replace(layout, cytassist=None), frame, scales, None, _options())
        assert set(images) == {"hires_image", "lowres_image", "full_image"}


class TestWarpGrid:
    """The regular output grid covers the complete finite projective footprint."""

    def test_horizon(self) -> None:
        matrix = np.eye(3)
        matrix[2] = [1, 0, -5]
        with pytest.raises(ReaderFormatError, match="horizon"):
            _warp_grid((10, 10), matrix)


class TestWarpCytassist:
    """Each tile exactly matches whole-array SciPy sampling without eager decoding."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_projective_matches_scipy(self, dtype: type[np.uint8 | np.uint16 | np.float32]) -> None:
        source = (np.arange(2 * 71 * 83).reshape(2, 71, 83) % 251).astype(dtype)
        h = np.array([[1.8, 0.3, 5], [-0.2, -1.9, 140], [0.0008, -0.0006, 1]])
        pixels, matrix = warp_cytassist(
            da.from_array(source, chunks=(1, 16, 16)), _identity_frame(), h, make_raster_options(chunks=(1, 16, 16))
        )
        output_to_source = h @ np.linalg.inv(np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)) @ matrix
        yy, xx = np.indices((int(pixels.shape[1]), int(pixels.shape[2])), dtype=float)
        coordinates = apply_projective(output_to_source, np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5))) - 0.5
        coordinates = np.stack((coordinates[:, 1], coordinates[:, 0])).reshape(2, *pixels.shape[1:])
        expected = np.stack(
            [
                map_coordinates(channel, coordinates, order=1, mode="constant", cval=0, prefilter=False)
                for channel in source
            ]
        )
        actual = pixels.compute(scheduler="synchronous")
        assert actual.dtype == source.dtype
        np.testing.assert_array_equal(actual, expected)
        restored = pickle.loads(pickle.dumps(pixels))  # noqa: S301, RUF100 - Test-owned lazy graph.
        np.testing.assert_array_equal(restored[:, :3, :4].compute(), expected[:, :3, :4])

    def test_physical_affine_ramp_at_pixel_centers(self) -> None:
        # Independent intensity oracle: the continuous field is f(gx,gy)=gx+3*gy.
        # Bilinear interpolation of an affine field must reproduce that field
        # at output pixel centres, regardless of rotation, translation or scale.
        yy, xx = np.indices((71, 83), dtype=float)
        global_x = 2 * (yy + 0.5) + 5
        global_y = -2 * (xx + 0.5) + 170
        intensity = global_x + 3 * global_y
        # Quarter-turn and dyadic scales make every expected intensity exact,
        # independently avoiding floating-point comparison tolerances.
        registration = np.array([[0, -1, 84.5], [1, 0, -2], [0, 0, 1]], dtype=float)
        pixels, output_matrix = warp_cytassist(
            da.from_array(intensity[None], chunks=(1, 16, 16)),
            _identity_frame(),
            registration,
            make_raster_options(chunks=(1, 16, 16)),
        )
        out_rows, out_cols = np.indices((int(pixels.shape[1]), int(pixels.shape[2])), dtype=float)
        gx = output_matrix[0, 0] * (out_cols + 0.5) + output_matrix[0, 2]
        gy = output_matrix[1, 1] * (out_rows + 0.5) + output_matrix[1, 2]
        source_x = (170 - gy) / 2
        source_y = (gx - 5) / 2
        interior = (source_x > 1) & (source_x < 82) & (source_y > 1) & (source_y < 70)
        actual = pixels.compute(scheduler="synchronous")[0]
        np.testing.assert_array_equal(actual[interior], (gx + 3 * gy)[interior])

    def test_tiled_tiff_partial_read(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = ArtifactFile(tmp_path / "cytassist.tiff")
        source = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
        tifffile.imwrite(path, source, tile=(16, 16), compression="deflate")
        spy = mocker.spy(tiff_dask, "_read_tiff_selection")
        raster = _normalize(read_tiff(inspect_tiff(path), series=0, level=0))
        # This registration makes CytAssist pixels equal global micrometres.
        h = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
        warped, _ = warp_cytassist(raster, _identity_frame(), h, make_raster_options(chunks=(1, 16, 16)))
        spy.assert_not_called()
        np.testing.assert_array_equal(
            warped[:, 33:35, 33:35].compute(scheduler="synchronous"), source[None, 33:35, 33:35]
        )
        assert spy.call_count < 9
        for call in spy.call_args_list:
            selection = call.args[1]
            assert all(part.stop - part.start <= 16 for part in selection)
