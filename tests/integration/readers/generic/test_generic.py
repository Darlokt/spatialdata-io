"""Integration tests for generic reader, converter, and CLI workflows."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import imagecodecs
import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata import SpatialData
from spatialdata.transformations import get_transformation
from xarray import DataArray, DataTree

from spatialdata_io._cli.commands.generic import generic_command
from spatialdata_io.converters.generic_to_zarr import generic_to_zarr
from spatialdata_io.readers._utils.errors import RasterFormatError
from spatialdata_io.readers._utils.image.tiff import _dask as tiff_dask
from spatialdata_io.readers.generic import generic, image

if TYPE_CHECKING:
    from pathlib import Path

    from click.testing import CliRunner
    from pytest_mock import MockerFixture

type AxisCase = tuple[str, tuple[int, ...], tuple[int, ...] | None, tuple[str, ...]]


def _write_tiff(
    path: Path,
    data: np.ndarray,
    axes: str,
    *,
    compression: str | None = None,
) -> None:
    tifffile.imwrite(
        path,
        data,
        compression=compression,
        photometric="minisblack",
        metadata={"axes": axes},
        tile=(16, 16),
        bigtiff=path.suffix.lower() == ".btf",
    )


def _write_geojson(path: Path) -> None:
    feature_collection = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"instance": 1},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                },
            }
        ],
    }
    path.write_text(
        json.dumps(feature_collection),
        encoding="utf-8",
    )


class TestImage:
    """Exercise exact lazy image-model composition."""

    @pytest.mark.parametrize("compression", [None, "deflate"])
    @pytest.mark.parametrize(
        "axis_case",
        [
            ("YX", (19, 23), None, ("c", "y", "x")),
            ("YXC", (19, 23, 3), (2, 0, 1), ("c", "y", "x")),
            ("ZYX", (2, 19, 23), None, ("c", "z", "y", "x")),
            ("XYCZ", (23, 19, 3, 2), (2, 3, 1, 0), ("c", "z", "y", "x")),
        ],
    )
    def test_preserves_values_and_normalizes_2d_and_3d_axes(
        self,
        tmp_path: Path,
        compression: str | None,
        axis_case: AxisCase,
    ) -> None:
        source_axes, shape, transpose, expected_dims = axis_case
        path = tmp_path / "image.tif"
        source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
        _write_tiff(path, source, source_axes, compression=compression)

        result = image(path, source_axes.lower(), "cells")

        expected = source if transpose is None else source.transpose(transpose)
        if "C" not in source_axes:
            expected = expected[None]
        assert isinstance(result, DataArray)
        assert result.dims == expected_dims
        assert result.dtype == source.dtype
        assert get_transformation(result, to_coordinate_system="cells") is not None
        np.testing.assert_array_equal(result.data.compute(), expected)

    def test_constructs_without_executing_pixel_tasks(self, tmp_path: Path) -> None:
        path = tmp_path / "lazy.btf"
        source = np.arange(3 * 19 * 23, dtype=np.uint16).reshape(3, 19, 23)
        _write_tiff(path, source, "CYX", compression="deflate")
        executed: list[object] = []

        with Callback(pretask=lambda key, *_: executed.append(key)):
            result = image(path, "cyx", "global")

        assert not executed
        assert result.chunksizes == {"c": (1, 1, 1), "y": (19,), "x": (23,)}

    def test_small_selection_does_not_decode_complete_large_tiff(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        path = tmp_path / "bounded.tif"
        source = np.arange(1100 * 1100, dtype=np.uint16).reshape(1100, 1100)
        tifffile.imwrite(
            path,
            source,
            compression="deflate",
            photometric="minisblack",
            metadata={"axes": "YX"},
            tile=(256, 256),
        )
        selection_spy = mocker.spy(tiff_dask, "_read_tiff_selection")
        result = image(path, "yx", "global")

        computed = result.data[0, :8, :8].compute(scheduler="synchronous")

        np.testing.assert_array_equal(computed, source[:8, :8])
        assert selection_spy.call_count == 16

    def test_selects_tiff_series(self, tmp_path: Path) -> None:
        path = tmp_path / "series.tif"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        result = image(path, "yx", "global", tiff_series=1)

        np.testing.assert_array_equal(result.data.compute(), second[None])

    def test_applies_custom_3d_chunks(self, tmp_path: Path) -> None:
        path = tmp_path / "volume.tif"
        source = np.arange(2 * 3 * 19 * 23, dtype=np.uint16).reshape(2, 3, 19, 23)
        _write_tiff(path, source, "CZYX", compression="deflate")

        result = generic(path, data_axes="czyx", chunks=(1, 2, 8, 9))

        assert isinstance(result, DataArray)
        assert result.chunksizes == {
            "c": (1, 1),
            "z": (2, 1),
            "y": (8, 8, 3),
            "x": (9, 9, 5),
        }
        np.testing.assert_array_equal(result.data.compute(), source)

    def test_default_3d_chunks_preserve_channel_and_depth_storage(self, tmp_path: Path) -> None:
        path = tmp_path / "volume.tif"
        source = np.arange(2 * 3 * 19 * 23, dtype=np.uint16).reshape(2, 3, 19, 23)
        _write_tiff(path, source, "CZYX", compression="deflate")

        result = image(path, "czyx", "global")

        assert isinstance(result, DataArray)
        assert result.chunksizes == {
            "c": (1, 1),
            "z": (1, 1, 1),
            "y": (19,),
            "x": (23,),
        }

    def test_selects_tiff_pyramid_level_before_model_pyramid(self, tmp_path: Path) -> None:
        path = tmp_path / "pyramid.btf"
        base = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
        reduced = base[::2, ::2]
        with tifffile.TiffWriter(path, bigtiff=True) as writer:
            writer.write(
                base,
                tile=(16, 16),
                subifds=1,
                photometric="minisblack",
                metadata={"axes": "YX"},
            )
            writer.write(reduced, tile=(16, 16), subfiletype=1, photometric="minisblack")

        result = image(path, "yx", "global", tiff_level=1, scale_factors=[2])

        assert isinstance(result, DataTree)
        base_level = result["scale0"]["image"]
        assert base_level.shape == (1, 16, 16)
        np.testing.assert_array_equal(base_level.data.compute(), reduced[None])

    @pytest.mark.parametrize("suffix", [".png", ".jpg"])
    def test_reads_encoded_images_lazily_and_exactly(self, tmp_path: Path, suffix: str) -> None:
        path = tmp_path / f"encoded{suffix}"
        source = np.arange(9 * 11 * 3, dtype=np.uint8).reshape(9, 11, 3)
        if suffix == ".png":
            path.write_bytes(imagecodecs.png_encode(source))
            expected = imagecodecs.png_decode(path.read_bytes())
        else:
            path.write_bytes(imagecodecs.jpeg8_encode(source, level=90))
            expected = imagecodecs.jpeg8_decode(path.read_bytes())
        executed: list[object] = []

        with Callback(pretask=lambda key, *_: executed.append(key)):
            result = image(path, "yxc", "global")

        assert not executed
        np.testing.assert_array_equal(result.data.compute(), expected.transpose(2, 0, 1))

    def test_rejects_suffix_signature_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong.png"
        path.write_bytes(imagecodecs.jpeg8_encode(np.zeros((5, 7), dtype=np.uint8)))

        with pytest.raises(RasterFormatError, match="PNG signature"):
            image(path, "yx", "global")


class TestGeneric:
    """Exercise the public mixed-format reader."""

    def test_reads_geojson_with_transformation(self, tmp_path: Path) -> None:
        path = tmp_path / "shapes.geojson"
        _write_geojson(path)

        result = generic(path, coordinate_system="cells")

        assert len(result) == 1
        assert get_transformation(result, to_coordinate_system="cells") is not None


class TestGenericToZarr:
    """Exercise silent library persistence."""

    def test_round_trips_exact_pixels_without_stdout(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        path = tmp_path / "image.png"
        output = tmp_path / "output.zarr"
        source = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        path.write_bytes(imagecodecs.png_encode(source))

        generic_to_zarr(path, output, data_axes="yxc")

        assert capsys.readouterr().out == ""
        spatial_data = SpatialData.read(output)
        result = cast("DataArray", spatial_data["image"])
        np.testing.assert_array_equal(result.data.compute(), source.transpose(2, 0, 1))

    def test_appends_geojson_and_rejects_duplicate_before_parsing(
        self,
        tmp_path: Path,
        mocker: MockerFixture,
    ) -> None:
        image_path = tmp_path / "image.png"
        shapes_path = tmp_path / "shapes.geojson"
        output = tmp_path / "output.zarr"
        image_path.write_bytes(imagecodecs.png_encode(np.zeros((5, 7), dtype=np.uint8)))
        _write_geojson(shapes_path)
        generic_to_zarr(image_path, output, data_axes="yx")

        generic_to_zarr(shapes_path, output, name="regions")

        spatial_data = SpatialData.read(output)
        assert set(spatial_data.images) == {"image"}
        assert set(spatial_data.shapes) == {"regions"}

        parse = mocker.patch("spatialdata_io.converters.generic_to_zarr.generic")
        with pytest.raises(ValueError, match="already exists"):
            generic_to_zarr(tmp_path / "missing.png", output, name="regions", data_axes="yx")
        parse.assert_not_called()


class TestReadGenericWrapper:
    """Exercise CLI validation and TIFF selection forwarding."""

    def test_writes_selected_tiff_series(
        self,
        tmp_path: Path,
        runner: CliRunner,
    ) -> None:
        path = tmp_path / "series.tif"
        output = tmp_path / "output.zarr"
        first = np.arange(35, dtype=np.uint8).reshape(5, 7)
        second = np.arange(99, dtype=np.uint16).reshape(9, 11)
        with tifffile.TiffWriter(path) as writer:
            writer.write(first, photometric="minisblack", metadata={"axes": "YX"})
            writer.write(second, photometric="minisblack", metadata={"axes": "YX"})

        result = runner.invoke(
            generic_command,
            [
                "--input",
                str(path),
                "--output",
                str(output),
                "--data-axes",
                "yx",
                "--tiff-series",
                "1",
                "--chunks",
                "1",
                "--chunks",
                "4",
                "--chunks",
                "5",
            ],
        )

        assert result.exit_code == 0, result.output
        assert f"Data written to {output}" in result.output
        spatial_data = SpatialData.read(output)
        stored = cast("DataArray", spatial_data["series"])
        assert stored.chunksizes == {"c": (1,), "y": (4, 4, 1), "x": (5, 5, 1)}
        np.testing.assert_array_equal(stored.data.compute(), second[None])

    def test_reports_invalid_axes_from_reader(self, tmp_path: Path, runner: CliRunner) -> None:
        path = tmp_path / "image.png"
        output = tmp_path / "output.zarr"
        path.write_bytes(imagecodecs.png_encode(np.zeros((5, 7), dtype=np.uint8)))

        result = runner.invoke(
            generic_command,
            ["--input", str(path), "--output", str(output), "--data-axes", "invalid"],
        )

        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert "data_axes" in str(result.exception)

    def test_appends_geojson_with_custom_name(self, tmp_path: Path, runner: CliRunner) -> None:
        image_path = tmp_path / "image.png"
        shapes_path = tmp_path / "shapes.geojson"
        output = tmp_path / "output.zarr"
        image_path.write_bytes(imagecodecs.png_encode(np.zeros((5, 7), dtype=np.uint8)))
        _write_geojson(shapes_path)
        create = runner.invoke(
            generic_command,
            ["--input", str(image_path), "--output", str(output), "--data-axes", "yx"],
        )
        assert create.exit_code == 0, create.output

        append = runner.invoke(
            generic_command,
            ["--input", str(shapes_path), "--output", str(output), "--name", "regions"],
        )

        assert append.exit_code == 0, append.output
        assert f"Element regions written to {output}" in append.output
        spatial_data = SpatialData.read(output)
        assert set(spatial_data.images) == {"image"}
        assert set(spatial_data.shapes) == {"regions"}
