"""CODEX layout, source schemas, identifier precision, and image options."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from xarray import DataArray, DataTree

from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.codex._images import _read_image, _validate_image_options
from tests.integration.readers.codex._fixtures import write_csv, write_image

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from spatialdata import SpatialData

DEFAULT_OPTIONS = make_raster_options()


class TestValidateImageOptions:
    """Reject invalid public options before expensive source work."""

    @pytest.mark.parametrize("series", [-1, True, 1.5])
    def test_invalid_series(self, series: float) -> None:
        with pytest.raises(ValueError, match="tiff_series"):
            _validate_image_options(None, series)  # type: ignore[bad-argument-type, arg-type]  # Untyped caller input.

    def test_invalid_axes_type(self) -> None:
        with pytest.raises(TypeError, match="data_axes"):
            _validate_image_options(["c", "y", "x"], None)  # type: ignore[bad-argument-type, arg-type]  # Untyped caller input.

    @pytest.mark.parametrize(
        "read",
        [
            partial(codex, chunks=8),
            partial(codex, scale_factors=(2,)),
            partial(codex, data_axes="CYX"),
            partial(codex, tiff_series=0),
        ],
        ids=["chunks", "scale_factors", "data_axes", "tiff_series"],
    )
    def test_nondefault_options_require_an_image(self, tmp_path: Path, read: Callable[[Path], SpatialData]) -> None:
        write_csv(tmp_path / "cells.csv")

        with pytest.raises(ValueError, match="require an image"):
            read(tmp_path)

    def test_default_raster_options_need_no_image(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")

        result = codex(tmp_path, scale_factors=(), chunks=(1, 1024, 1024))

        assert dict(result.images) == {}

    def test_invalid_raster_option_fails_before_discovery(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="scale_factors"):
            codex(tmp_path / "missing", scale_factors=(1,))


class TestReadImage:
    """Keep explicit channel semantics, TIFF axes, and small images lazy."""

    def test_malformed_tiff_has_reader_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"invalid TIFF")

        with pytest.raises(ValueError, match="TIFF") as caught:
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        assert any("codex" in note and "stitched image" in note for note in caught.value.__notes__)

    def test_large_image_is_not_pyramided_automatically(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = write_image(path)

        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        assert isinstance(result, DataArray)
        # Default tiles exceed the image, so each chunk is one whole channel.
        assert result.data.chunksize == (1, 32, 40)
        np.testing.assert_array_equal(result.data.compute(), image)

    def test_scale_factors_build_a_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = write_image(path)

        result = _read_image(
            ArtifactFile(path), data_axes=None, tiff_series=None, raster=make_raster_options((2, 2), (1, 8, 8))
        )

        assert isinstance(result, DataTree)
        assert list(result.children) == ["scale0", "scale1", "scale2"]
        scale0 = result["scale0"]["image"]
        assert scale0.data.chunksize == (1, 8, 8)
        np.testing.assert_array_equal(scale0.data.compute(), image)
        assert result["scale1"]["image"].shape == (2, 16, 20)
        assert result["scale2"]["image"].shape == (2, 8, 10)

    def test_small_grayscale_is_single_scale(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.ones((2, 3), dtype=np.uint8), metadata={"axes": "YX"})

        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        assert isinstance(result, DataArray)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result.data.compute(), np.ones((1, 2, 3), dtype=np.uint8))

    @pytest.mark.parametrize("channels", [3, 4, 5])
    def test_arbitrary_channels_are_not_rgb(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((channels, 8, 10), dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack"
        )

        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        assert result.coords["c"].to_numpy().tolist() == list(range(channels))

    def test_rgb_samples_are_ordered_channels(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        tifffile.imwrite(path, image, photometric="rgb")

        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        np.testing.assert_array_equal(result.data.compute(), image.transpose(2, 0, 1))

    def test_unknown_stack_axes_need_override(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((2, 8, 10), dtype=np.uint16), metadata={"axes": "QYX"}, photometric="minisblack"
        )

        with pytest.raises(ValueError, match="Axis 'q'"):
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, raster=DEFAULT_OPTIONS)

        result = _read_image(ArtifactFile(path), data_axes="CYX", tiff_series=None, raster=DEFAULT_OPTIONS)

        assert result.shape == (2, 8, 10)

    def test_invalid_declared_axes(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        write_image(path)

        with pytest.raises(ValueError, match="duplicate"):
            _read_image(ArtifactFile(path), data_axes="YYX", tiff_series=None, raster=DEFAULT_OPTIONS)
