"""CODEX layout, source schemas, identifier precision, and image options."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from xarray import DataArray

from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.codex import codex
from spatialdata_io.readers.codex._images import _read_image, _validate_image_options
from tests.integration.readers.codex._fixtures import write_csv, write_image

if TYPE_CHECKING:
    from pathlib import Path


class TestValidateImageOptions:
    """Reject invalid public options before expensive source work."""

    @pytest.mark.parametrize("series", [-1, True, 1.5])
    def test_invalid_series(self, series: float) -> None:
        with pytest.raises(ValueError, match="tiff_series"):
            _validate_image_options(None, series, None)  # type: ignore[bad-argument-type, arg-type]  # Untyped caller input.

    @pytest.mark.parametrize("chunks", [0, -1, (1, 0, 1), (1, 2), (), True, (1, 2.0, 3), "auto"])
    def test_invalid_chunks(self, chunks: int | tuple[int | float, ...] | str) -> None:
        with pytest.raises((TypeError, ValueError), match="chunks"):
            _validate_image_options(None, None, chunks)  # type: ignore[bad-argument-type, arg-type]  # Untyped caller input.

    def test_invalid_axes_type(self) -> None:
        with pytest.raises(TypeError, match="data_axes"):
            _validate_image_options(["c", "y", "x"], None, None)  # type: ignore[bad-argument-type, arg-type]  # Untyped caller input.

    def test_options_require_an_image(self, tmp_path: Path) -> None:
        write_csv(tmp_path / "cells.csv")
        with pytest.raises(ValueError, match="require an image"):
            codex(tmp_path, chunks=8)


class TestReadImage:
    """Keep explicit channel semantics, TIFF axes, and small images lazy."""

    def test_malformed_tiff_has_reader_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        path.write_bytes(b"invalid TIFF")
        with pytest.raises(ValueError, match="TIFF") as caught:
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, chunks=None)
        assert any("codex" in note and "stitched image" in note for note in caught.value.__notes__)

    def test_small_grayscale_is_single_scale(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.ones((2, 3), dtype=np.uint8), metadata={"axes": "YX"})
        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, chunks=None)
        assert isinstance(result, DataArray)
        assert result.shape == (1, 2, 3)
        np.testing.assert_array_equal(result.data.compute(), np.ones((1, 2, 3), dtype=np.uint8))

    @pytest.mark.parametrize("channels", [3, 4, 5])
    def test_arbitrary_channels_are_not_rgb(self, tmp_path: Path, channels: int) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((channels, 8, 10), dtype=np.uint16), metadata={"axes": "CYX"}, photometric="minisblack"
        )
        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, chunks=None)
        assert result["scale0"]["image"].coords["c"].to_numpy().tolist() == list(range(channels))

    def test_rgb_samples_are_ordered_channels(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        image = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        tifffile.imwrite(path, image, photometric="rgb")
        result = _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, chunks=None)
        np.testing.assert_array_equal(result["scale0"]["image"].data.compute(), image.transpose(2, 0, 1))

    def test_unknown_stack_axes_need_override(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(
            path, np.zeros((2, 8, 10), dtype=np.uint16), metadata={"axes": "QYX"}, photometric="minisblack"
        )
        with pytest.raises(ValueError, match="Axis 'q'"):
            _read_image(ArtifactFile(path), data_axes=None, tiff_series=None, chunks=None)
        result = _read_image(ArtifactFile(path), data_axes="CYX", tiff_series=None, chunks=None)
        assert result["scale0"]["image"].shape == (2, 8, 10)

    def test_invalid_declared_axes(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        write_image(path)
        with pytest.raises(ValueError, match="duplicate"):
            _read_image(ArtifactFile(path), data_axes="YYX", tiff_series=None, chunks=None)
