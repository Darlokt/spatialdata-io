"""Test DBiT image options, axes, channel interpretation, and pyramids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import imagecodecs
import numpy as np
import pytest
import tifffile
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import inspect_tiff, make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.dbit._images import _read_image, _validate_image_options

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture

DEFAULT_OPTIONS = make_raster_options()


class TestValidateImageOptions:
    """Reject malformed caller options before loading metadata or expression."""

    @pytest.mark.parametrize("axes", ["", "YYX", "CYXS", "C YX", "YX1", "CY", "\u03b1yx"])
    def test_invalid_axes(self, axes: str) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            _validate_image_options(axes, 0, 0)

    def test_invalid_axes_type(self) -> None:
        with pytest.raises(TypeError, match="data_axes"):
            _validate_image_options(["y", "x"], 0, 0)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    @pytest.mark.parametrize("value", [-1, True, 0.5, None])
    @pytest.mark.parametrize("option", ["series", "level"])
    def test_invalid_tiff_selection(self, value: float | None, option: str) -> None:
        with pytest.raises(ValueError, match=f"tiff_{option}"):
            _validate_image_options(None, value if option == "series" else 0, value if option == "level" else 0)  # pyrefly: ignore[bad-argument-type] - Untrusted caller input.

    @pytest.mark.parametrize("axes", [None, "YX", "YXS", "TZCYX"])
    def test_valid_options(self, axes: str | None) -> None:
        _validate_image_options(axes, 1, 2)


class TestReadImage:
    """Preserve scientific channels and named singleton dimensions lazily."""

    @pytest.mark.parametrize("shape", [(1, 1), (1, 6), (7, 1)])
    def test_grayscale_preserves_spatial_singletons(self, tmp_path: Path, shape: tuple[int, int]) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        tifffile.imwrite(path, source, metadata={"axes": "YX"})

        image = _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.dims == ("c", "y", "x")
        assert image.shape == (1, *shape)
        np.testing.assert_array_equal(image.data.compute(), source[None])

    @pytest.mark.parametrize("channels", [1, 3, 4, 5])
    def test_scientific_channels_keep_positional_labels(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(channels * 30, dtype=np.uint16).reshape(channels, 5, 6)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "CYX"})

        image = _read_image(
            ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=make_raster_options(chunks=2)
        )

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == list(range(channels))
        assert image.data.chunksize == (min(channels, 2), 2, 2)
        np.testing.assert_array_equal(image.data.compute(), source)

    @pytest.mark.parametrize("channels", [3, 4])
    def test_rgb_samples_keep_ordered_positional_labels(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(30 * channels, dtype=np.uint8).reshape(5, 6, channels)
        tifffile.imwrite(path, source, photometric="rgb")

        image = _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == list(range(channels))
        np.testing.assert_array_equal(image.data.compute(), source.transpose(2, 0, 1))

    @pytest.mark.parametrize("names", [["marker-b", "marker-a"], ["same", "same"], [None, None]])
    def test_ome_channel_names_or_unique_ids(self, tmp_path: Path, names: list[str | None]) -> None:
        path = tmp_path / "image.ome.tif"
        source = np.arange(60, dtype=np.uint16).reshape(1, 1, 2, 5, 6)
        tifffile.imwrite(
            path,
            source,
            photometric="minisblack",
            metadata={"axes": "TZCYX", "Channel": {"Name": names}},
        )

        image = _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        assert image.coords["c"].to_numpy().tolist() == (
            names if names == ["marker-b", "marker-a"] else ["Channel:0:0", "Channel:0:1"]
        )
        np.testing.assert_array_equal(image.data.compute(), source[0, 0])

    def test_unknown_stack_requires_explicit_axes(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        source = np.ones((2, 5, 6), dtype=np.uint8)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "QYX"})

        with pytest.raises(ValueError, match="Axis 'q'"):
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        image = _read_image(ArtifactFile(path), data_axes="CYX", tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        np.testing.assert_array_equal(image.data.compute(), source)

    def test_default_options_are_single_scale_channel_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        source = np.arange(3 * 40 * 50, dtype=np.uint16).reshape(3, 40, 50)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "CYX"}, tile=(16, 16))

        image = _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert isinstance(image, DataArray)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert image.data.chunks == ((1, 1, 1), (40,), (50,))
        np.testing.assert_array_equal(image.data.compute(), source)

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "image.png"
        source = np.arange(3 * 8 * 12, dtype=np.uint8).reshape(8, 12, 3)
        path.write_bytes(imagecodecs.png_encode(source))

        image = _read_image(
            ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=make_raster_options((2,), 4)
        )

        assert isinstance(image, DataTree)
        assert list(image.children) == ["scale0", "scale1"]
        scale0 = image["scale0"]["image"]
        assert scale0.data.chunks == ((3,), (4, 4), (4, 4, 4))
        np.testing.assert_array_equal(scale0.data.compute(), source.transpose(2, 0, 1))
        assert image["scale1"]["image"].shape == (3, 4, 6)

    def test_inconsistent_ome_channels_are_rejected(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "image.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((2, 5, 6), dtype=np.uint16),
            photometric="minisblack",
            metadata={"axes": "CYX", "Channel": {"Name": ["a", "b"]}},
        )

        metadata = inspect_tiff(path)
        assert metadata.ome is not None
        metadata.ome.images[0].pixels.channels.pop()
        mocker.patch("spatialdata_io.readers.dbit._images.inspect_tiff", return_value=metadata)

        with pytest.raises(ReaderFormatError, match="OME channel records"):
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

    @pytest.mark.parametrize(("series", "level"), [(1, 0), (0, 1)])
    def test_out_of_range_selection(self, tmp_path: Path, series: int, level: int) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.ones((5, 6), dtype=np.uint8))

        with pytest.raises(IndexError, match="index") as caught:
            _read_image(
                ArtifactFile(path), data_axes=None, tiff_series=series, tiff_level=level, raster=DEFAULT_OPTIONS
            )

        assert any("dbit" in note for note in caught.value.__notes__)

    @pytest.mark.parametrize("suffix", [".tif", ".png", ".jpg"])
    def test_malformed_image_has_reader_context(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"image{suffix}"
        path.write_bytes(b"invalid image")

        with pytest.raises(ValueError, match=r"TIFF|PNG|JPEG") as caught:
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=0, tiff_level=0, raster=DEFAULT_OPTIONS)

        assert any("dbit" in note and "histology image" in note for note in caught.value.__notes__)

    @pytest.mark.parametrize(("series", "level"), [(1, 0), (0, 1)])
    def test_encoded_images_reject_tiff_options(self, tmp_path: Path, series: int, level: int) -> None:
        with pytest.raises(ValueError, match="TIFF-family"):
            _read_image(
                ArtifactFile(tmp_path / "image.png"),
                data_axes=None,
                tiff_series=series,
                tiff_level=level,
                raster=DEFAULT_OPTIONS,
            )

    def test_encoded_images_reject_axis_override(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            _read_image(
                ArtifactFile(tmp_path / "image.png"),
                data_axes="YXC",
                tiff_series=0,
                tiff_level=0,
                raster=DEFAULT_OPTIONS,
            )
