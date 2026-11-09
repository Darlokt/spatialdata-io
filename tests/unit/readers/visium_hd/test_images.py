"""Lazy image placement, selection and bounded exact CytAssist sampling."""

from __future__ import annotations

import pickle
import re
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

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import (
    inspect_tiff,
    make_raster_options,
    normalize_cyx_image,
    read_tiff,
)
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.visium_hd._images import (
    ImageOptions,
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
    read_bin_scale_factors,
    read_source_metadata,
)
from spatialdata_io.readers.visium_hd._positions import fit_slide_frame
from spatialdata_io.readers.visium_hd._transforms import SlideFrame, apply_projective
from tests.integration.readers.visium_hd._fixtures import VERSION, _write_png, write_bundle

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Literal

    from pytest_mock import MockerFixture


_CONTEXT = ReaderErrorContext("visium_hd", "test image", VERSION)


def _options(*, data_axes: str | None = None, tiff_series: int | None = None) -> ImageOptions:
    return make_image_options(data_axes=data_axes, tiff_series=tiff_series, scale_factors=(), chunks=(1, 16, 16))


def _identity_frame() -> SlideFrame:
    return SlideFrame(np.eye(3), np.eye(3), np.eye(3), 0.03, 0.01)


def _bundle(root: Path) -> tuple[ImageLayout, SlideFrame, ScaleFactors, SourceMetadata]:
    write_bundle(root, images=True, fullres="root")
    layout = discover_layout(root, include_cytassist_image=True)
    assert layout.images is not None
    assert layout.feature_slice is not None
    return (
        layout.images,
        fit_slide_frame(layout.base_positions, (17, 17), version=VERSION),
        read_bin_scale_factors(layout.base_scale_factors, 2, version=VERSION),
        read_source_metadata(layout.feature_slice, version=VERSION),
    )


def _channels(image: DataArray | DataTree) -> list[object]:
    array = image if isinstance(image, DataArray) else image["scale0"]["image"]
    assert isinstance(array, DataArray)
    return array.coords["c"].to_numpy().tolist()


def _fullres(path: Path, options: ImageOptions | None = None) -> DataArray:
    """Read one full-resolution image through ``read_images`` with an identity registration."""
    layout = ImageLayout(None, None, None, ArtifactFile(path))
    scales = ScaleFactors(hires=1.0, lowres=1.0, microns_per_pixel=1.0)
    image = read_images(layout, _identity_frame(), scales, None, options or _options(), version=VERSION)["full_image"]
    assert isinstance(image, DataArray)
    return image


class TestMakeImageOptions:
    """Public raster options are validated and owned before opening files."""

    def test_owned_axes_and_factors(self) -> None:
        factors = [2]
        result = make_image_options(data_axes="CyX", tiff_series=0, scale_factors=factors, chunks=8)
        factors[0] = 4
        assert result.data_axes == "cyx"
        assert result.tiff_series == 0
        assert result.raster.scale_factors == (2,)

    @pytest.mark.parametrize("axes", ["", "YY", "CSYX", "CZX", "Y1X", "YXÄ"])
    def test_invalid_axes(self, axes: str) -> None:
        message = (
            "data_axes must contain distinct single-letter axes including X and Y, "
            f"with S (samples) counting as C; found {axes!r}."
        )
        with pytest.raises(ValueError, match=re.escape(message)):
            make_image_options(data_axes=axes, tiff_series=None, scale_factors=(), chunks=8)

    @pytest.mark.parametrize("axes", [17, ["C", "Y", "X"], ("Y", "X"), b"YX"])
    def test_axes_must_be_one_string(self, axes: object) -> None:
        message = f"data_axes must be a string such as 'CYX', or None; found {type(axes).__name__}."
        with pytest.raises(TypeError, match=re.escape(message)):
            make_image_options(data_axes=axes, tiff_series=None, scale_factors=(), chunks=8)  # pyrefly: ignore[bad-argument-type] - Runtime option validation.

    @pytest.mark.parametrize("series", [True, 1.5, "0"])
    def test_series_type(self, series: object) -> None:
        with pytest.raises(TypeError, match=r"tiff_series must be an integer or None; found"):
            make_image_options(data_axes=None, tiff_series=series, scale_factors=(), chunks=8)  # pyrefly: ignore[bad-argument-type] - Runtime option validation.

    def test_series_value(self) -> None:
        with pytest.raises(ValueError, match=r"tiff_series must be nonnegative; found -1\."):
            make_image_options(data_axes=None, tiff_series=-1, scale_factors=(), chunks=8)


class TestReadFullres:
    """TIFF family recognition uses content; encoded sources remain shared."""

    def test_tiff_content_and_series(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "unexpected.bin")
        tifffile.imwrite(path, np.ones((8, 9), np.uint16), bigtiff=True)
        image = _read_fullres(path, _options(), version=VERSION)
        np.testing.assert_array_equal(image.pixels.compute(), np.ones((1, 8, 9), np.uint16))
        assert image.channels is None

    def test_ambiguous_series_requires_selection(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "multiple.tif")
        with tifffile.TiffWriter(path) as writer:
            writer.write(np.ones((8, 9), np.uint16))
            writer.write(np.ones((5, 6), np.uint16))
        with pytest.raises(ReaderFormatError, match="found 2 TIFF series") as error:
            _read_fullres(path, _options(), version=VERSION)
        assert error.value.context == ReaderErrorContext("visium_hd", "full-resolution image", VERSION)
        assert error.value.path == path
        assert error.value.expected == "one TIFF series or an explicit tiff_series selection"
        assert _read_fullres(path, _options(tiff_series=1), version=VERSION).pixels.shape == (1, 5, 6)

    def test_out_of_range_series_has_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.tif")
        tifffile.imwrite(path, np.ones((8, 9), np.uint16))
        with pytest.raises(IndexError) as error:
            _read_fullres(path, _options(tiff_series=1), version=VERSION)
        assert any("full-resolution image" in note and VERSION in note for note in error.value.__notes__)

    def test_png_and_jpeg(self, tmp_path: Path) -> None:
        png = ArtifactFile(tmp_path / "image.png")
        _write_png(png, np.zeros((8, 9, 3), np.uint8))
        assert _read_fullres(png, _options(), version=VERSION).pixels.shape == (3, 8, 9)
        jpeg = ArtifactFile(tmp_path / "image.jpg")
        jpeg.write_bytes(imagecodecs.jpeg_encode(np.zeros((8, 9, 3), np.uint8)))
        assert _read_fullres(jpeg, _options(), version=VERSION).pixels.shape == (3, 8, 9)
        for options in (_options(tiff_series=0), _options(data_axes="YXC")):
            with pytest.raises(ValueError, match="only to TIFF"):
                _read_fullres(png, options, version=VERSION)

    def test_unknown_content(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.xyz")
        path.write_bytes(b"unknown")
        with pytest.raises(ReaderFormatError, match=r"no TIFF signature and suffix '\.xyz'") as error:
            _read_fullres(path, _options(), version=VERSION)
        assert error.value.context.detected_version == VERSION

    def test_unreadable_content_has_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "missing.tif")
        with pytest.raises(FileNotFoundError) as error:
            _read_fullres(path, _options(), version=VERSION)
        assert any("full-resolution image" in note for note in error.value.__notes__)

    def test_pyramid_guard(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.tif")
        tifffile.imwrite(path, np.ones((8, 9), np.uint16))
        options = replace(_options(), raster=make_raster_options((2, 2, 2, 2), (1, 16, 16)))
        with pytest.raises(ValueError, match=r"exceed the spatial extent of the full-resolution image \(8 x 9"):
            _read_fullres(path, options, version=VERSION)

    # The shared CYX normalization tests every axis case; these check the reader's context and path.
    def test_unmappable_declared_axes_are_malformed_content(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.tif")
        tifffile.imwrite(path, np.ones((2, 5, 7), np.uint16), photometric="minisblack", metadata={"axes": "ZYX"})
        with pytest.raises(ReaderFormatError, match="Axis 'z' has length 2") as error:
            _read_fullres(path, _options(), version=VERSION)
        assert error.value.context == ReaderErrorContext("visium_hd", "full-resolution image", VERSION)
        assert error.value.path == path
        assert type(error.value.__cause__) is ValueError

    def test_unmappable_data_axes_are_a_caller_error_with_context(self, tmp_path: Path) -> None:
        path = ArtifactFile(tmp_path / "image.tif")
        tifffile.imwrite(path, np.ones((2, 5, 7), np.uint16), photometric="minisblack", metadata={"axes": "CYX"})
        with pytest.raises(ValueError, match="rank") as error:
            _read_fullres(path, _options(data_axes="yx"), version=VERSION)
        assert type(error.value) is ValueError
        context = ReaderErrorContext("visium_hd", "full-resolution image", VERSION)
        assert error.value.__notes__ == [f"While handling {context.describe(path=path)}."]


class TestChannelNames:
    """Channels are RGB only when declared; otherwise OME names or positions."""

    @pytest.mark.parametrize(
        ("photometric", "extrasamples", "ome", "expected"),
        [
            ("rgb", None, False, ["r", "g", "b"]),
            ("rgb", ("unassalpha",), False, ["r", "g", "b", "a"]),
            ("minisblack", None, False, [0, 1, 2]),
            ("rgb", None, True, ["r", "g", "b"]),
        ],
        ids=["rgb", "rgba", "minisblack-samples", "ome-rgb"],
    )
    def test_tiff_samples_are_rgb_only_when_declared(
        self,
        tmp_path: Path,
        photometric: Literal["rgb", "minisblack"],
        extrasamples: tuple[str, ...] | None,
        expected: list[object],
        *,
        ome: bool,
    ) -> None:
        path = tmp_path / "full.tif"
        samples = 3 if extrasamples is None else 4
        tifffile.imwrite(
            path,
            np.zeros((5, 7, samples), dtype=np.uint8),
            photometric=photometric,
            planarconfig="contig",
            extrasamples=extrasamples,
            ome=ome,
        )
        assert _channels(_fullres(path)) == expected

    @pytest.mark.parametrize("ome", [True, False])
    def test_four_channel_fluorescence_is_never_rgba(self, tmp_path: Path, *, ome: bool) -> None:
        path = tmp_path / "full.ome.tif"
        names = ["DAPI", "CD3", "CD8", "PanCK"]
        values = np.arange(4 * 5 * 7, dtype=np.uint16).reshape(4, 5, 7)
        metadata: dict[str, object] = {"axes": "CYX"}
        if ome:
            metadata["Channel"] = {"Name": names}
        tifffile.imwrite(path, values, ome=ome, photometric="minisblack", metadata=metadata)
        image = _fullres(path)
        assert _channels(image) == (names if ome else [0, 1, 2, 3])
        assert image.dtype == np.dtype(np.uint16)
        np.testing.assert_array_equal(image.data.compute(), values)

    def test_single_channel_ome_tiff_keeps_its_name(self, tmp_path: Path) -> None:
        path = tmp_path / "full.ome.tif"
        tifffile.imwrite(
            path, np.zeros((5, 7), dtype=np.uint16), ome=True, metadata={"axes": "YX", "Channel": {"Name": ["DAPI"]}}
        )
        assert _channels(_fullres(path)) == ["DAPI"]

    def test_axes_reinterpreted_as_channels_get_positions(self, tmp_path: Path) -> None:
        path = tmp_path / "full.ome.tif"
        tifffile.imwrite(path, np.zeros((3, 5, 7), dtype=np.uint16), ome=True, metadata={"axes": "ZYX"})
        image = _fullres(path, _options(data_axes="CYX"))
        assert image.shape == (3, 5, 7)
        assert _channels(image) == [0, 1, 2]

    def test_rgb_jpeg_and_gray_png(self, tmp_path: Path) -> None:
        jpeg = tmp_path / "full.jpg"
        jpeg.write_bytes(imagecodecs.jpeg_encode(np.zeros((5, 7, 3), dtype=np.uint8)))
        assert _channels(_fullres(jpeg)) == ["r", "g", "b"]
        png = tmp_path / "full.png"
        png.write_bytes(imagecodecs.png_encode(np.zeros((5, 7), dtype=np.uint8)))
        assert _channels(_fullres(png)) == [0]

    def test_space_ranger_pngs_and_cytassist(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        rgba = tmp_path / "rgba.png"
        rgba.write_bytes(imagecodecs.png_encode(np.zeros((24, 24, 4), dtype=np.uint8)))
        images = read_images(
            replace(layout, lowres=ArtifactFile(rgba)), frame, scales, source, _options(), version=VERSION
        )
        assert _channels(images["hires_image"]) == ["r", "g", "b"]
        assert _channels(images["lowres_image"]) == ["r", "g", "b", "a"]
        assert _channels(images["full_image"]) == ["r", "g", "b"]
        assert _channels(images["cytassist_image"]) == ["r", "g", "b"]

    def test_fluorescence_cytassist_is_not_rgb(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.cytassist is not None
        tifffile.imwrite(
            layout.cytassist, np.zeros((3, 48, 48), dtype=np.uint16), photometric="minisblack", metadata={"axes": "CYX"}
        )
        images = read_images(replace(layout, fullres=None), frame, scales, source, _options(), version=VERSION)
        assert _channels(images["cytassist_image"]) == [0, 1, 2]


class TestReadImages:
    """Microscopy keeps source pixels and exact scales; CytAssist resamples lazily."""

    def test_pixels_dtypes_affines_and_zero_tasks(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        tasks: list[object] = []
        with Callback(pretask=lambda key, _dsk, _state: tasks.append(key)):
            images = read_images(layout, frame, scales, source, _options(), version=VERSION)
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
        images = read_images(layout, frame, scales, source, options, version=VERSION)
        assert isinstance(images["full_image"], DataTree)
        assert isinstance(images["hires_image"], DataArray)
        assert isinstance(images["cytassist_image"], DataArray)

    def test_missing_images_and_absent_fullres_options(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        empty = replace(layout, fullres=None, hires=None, lowres=None, cytassist=None)
        assert read_images(empty, frame, scales, source, _options(), version=VERSION) == {}
        with pytest.raises(ValueError, match="require a full-resolution"):
            read_images(empty, frame, scales, source, _options(tiff_series=0), version=VERSION)

    def test_longest_side_mismatch(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.fullres is not None
        tifffile.imwrite(layout.fullres, np.zeros((98, 90, 3), np.uint16), photometric="rgb")
        with pytest.raises(ReaderFormatError, match="longest sides"):
            read_images(layout, frame, scales, source, _options(), version=VERSION)

    def test_short_side_can_differ_and_fullres_without_hires(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.fullres is not None
        tifffile.imwrite(layout.fullres, np.zeros((96, 89, 3), np.uint16), photometric="rgb")
        images = read_images(layout, frame, scales, source, _options(), version=VERSION)
        assert images["full_image"].shape == (3, 96, 89)
        assert "full_image" in read_images(
            replace(layout, hires=None), frame, scales, source, _options(), version=VERSION
        )

    @pytest.mark.parametrize("missing_metadata", [False, True])
    def test_absent_cytassist_transform(self, tmp_path: Path, *, missing_metadata: bool) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        selected_source = None if missing_metadata else replace(source, cytassist=None)
        with pytest.raises(ReaderFormatError, match="null CytAssist"):
            read_images(layout, frame, scales, selected_source, _options(), version=VERSION)

    def test_ambiguous_cytassist_series(self, tmp_path: Path) -> None:
        layout, frame, scales, source = _bundle(tmp_path)
        assert layout.cytassist is not None
        with tifffile.TiffWriter(layout.cytassist) as writer:
            writer.write(np.ones((8, 9), np.uint8))
            writer.write(np.ones((5, 6), np.uint8))
        with pytest.raises(ReaderFormatError, match="expected one TIFF series; found 2 TIFF series") as error:
            read_images(layout, frame, scales, source, _options(), version=VERSION)
        assert error.value.context == ReaderErrorContext("visium_hd", "CytAssist image", VERSION)

    def test_microscopy_without_feature_slice(self, tmp_path: Path) -> None:
        layout, frame, scales, _ = _bundle(tmp_path)
        images = read_images(replace(layout, cytassist=None), frame, scales, None, _options(), version=VERSION)
        assert set(images) == {"hires_image", "lowres_image", "full_image"}


class TestWarpGrid:
    """The regular output grid covers the complete finite projective footprint."""

    def test_horizon(self) -> None:
        matrix = np.eye(3)
        matrix[2] = [1, 0, -5]
        with pytest.raises(ReaderFormatError, match="horizon"):
            _warp_grid((10, 10), matrix, context=_CONTEXT)


class TestWarpCytassist:
    """Each tile exactly matches whole-array SciPy sampling without eager decoding."""

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
    def test_projective_matches_scipy(self, dtype: type[np.uint8 | np.uint16 | np.float32]) -> None:
        source = (np.arange(2 * 71 * 83).reshape(2, 71, 83) % 251).astype(dtype)
        h = np.array([[1.8, 0.3, 5], [-0.2, -1.9, 140], [0.0008, -0.0006, 1]])
        pixels, matrix = warp_cytassist(
            da.from_array(source, chunks=(1, 16, 16)),
            _identity_frame(),
            h,
            make_raster_options(chunks=(1, 16, 16)),
            context=_CONTEXT,
        )
        output_to_source = h @ np.linalg.inv(np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)) @ matrix
        yy, xx = np.indices((int(pixels.shape[1]), int(pixels.shape[2])), dtype=float)
        coordinates = (
            apply_projective(output_to_source, np.column_stack((xx.ravel() + 0.5, yy.ravel() + 0.5)), context=_CONTEXT)
            - 0.5
        )
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
            context=_CONTEXT,
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
        raster, _ = normalize_cyx_image(
            read_tiff(inspect_tiff(path, context=_CONTEXT), series=0, level=0, context=_CONTEXT),
            None,
            tiff=None,
            context=_CONTEXT,
            path=path,
        )
        # This registration makes CytAssist pixels equal global micrometres.
        h = np.array([[2, 0, 1], [0, 2, 1], [0, 0, 1]], dtype=float)
        warped, _ = warp_cytassist(
            raster, _identity_frame(), h, make_raster_options(chunks=(1, 16, 16)), context=_CONTEXT
        )
        spy.assert_not_called()
        np.testing.assert_array_equal(
            warped[:, 33:35, 33:35].compute(scheduler="synchronous"), source[None, 33:35, 33:35]
        )
        assert spy.call_count < 9
        for call in spy.call_args_list:
            selection = call.args[1]
            assert all(part.stop - part.start <= 16 for part in selection)
