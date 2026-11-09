"""Unit tests for generic image loading and model construction."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import dask.array as da
import numpy as np
import pytest
import tifffile

from spatialdata_io.readers._utils.errors import ReaderErrorContext, ReaderFormatError
from spatialdata_io.readers._utils.image import LazyRaster
from spatialdata_io.readers.generic import image

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from spatialdata_io.readers.generic._images import Chunks


class TestImage:
    """Validate axes, dispatch, chunks, and lazy model construction."""

    @pytest.mark.parametrize("coordinate_system", ["", " ", "\t\n"])
    def test_rejects_blank_coordinate_system_before_reading(self, coordinate_system: str) -> None:
        with pytest.raises(ValueError, match="coordinate_system"):
            image("missing.tif", "yx", coordinate_system)

    @pytest.mark.parametrize("coordinate_system", [None, 3])
    def test_rejects_nonstring_coordinate_system_before_reading(self, coordinate_system: object) -> None:
        with pytest.raises(TypeError, match="coordinate_system"):
            image("missing.tif", "yx", cast("str", coordinate_system))

    def test_rejects_ome_spatial_dimension_disagreeing_with_base_raster(self, tmp_path: Path) -> None:
        path = tmp_path / "contradictory.ome.tif"
        tifffile.imwrite(
            path,
            np.zeros((16, 24), dtype=np.uint16),
            photometric="minisblack",
            metadata={"axes": "YX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 2.0},
        )
        description = tifffile.tiffcomment(path)
        assert isinstance(description, str)
        tifffile.tiffcomment(path, description.replace('SizeX="24"', 'SizeX="25"'))

        # tifffile falls back to the real raster shape; Generic owns rejecting
        # the resulting contradiction with the still-declared OME dimensions.
        with pytest.raises(ReaderFormatError) as error:
            image(path, "yx", "sample")

        assert error.value.context.reader == "generic"
        assert error.value.context.artifact == "image"
        assert error.value.path == path
        assert error.value.found == "OME X dimension 25 disagrees with its base raster"
        assert error.value.expected == "matching source and OME spatial dimensions"

    @pytest.mark.parametrize(
        ("source_axes", "shape", "expected_dims", "expected_shape"),
        [
            ("yx", (5, 7), ("c", "y", "x"), (1, 5, 7)),
            ("yxc", (5, 7, 3), ("c", "y", "x"), (3, 5, 7)),
            ("zyx", (2, 5, 7), ("c", "z", "y", "x"), (1, 2, 5, 7)),
            ("xyzc", (7, 5, 2, 3), ("c", "z", "y", "x"), (3, 2, 5, 7)),
        ],
    )
    def test_normalizes_supported_axes_without_computing_source(
        self,
        mocker: MockerFixture,
        source_axes: str,
        shape: tuple[int, ...],
        expected_dims: tuple[str, ...],
        expected_shape: tuple[int, ...],
    ) -> None:
        data = da.zeros(shape, chunks=shape, dtype=np.uint16)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=tuple(source_axes)), None),
        )

        result = image("image.tif", source_axes.upper(), "global")

        assert result.dims == expected_dims
        assert result.shape == expected_shape

    def test_default_2d_chunks_hold_one_channel_with_square_tiles(self, mocker: MockerFixture) -> None:
        data = da.zeros((7, 5, 3), chunks=(7, 5, 3), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("y", "x", "c")), None),
        )
        parse = mocker.patch(
            "spatialdata_io.readers.generic._images.Image2DModel.parse",
            return_value=mocker.sentinel.image,
        )

        result = image("image.png", "yxc", "global")

        assert result is mocker.sentinel.image
        assert parse.call_args.kwargs["dims"] == ("c", "y", "x")
        assert parse.call_args.kwargs["chunks"] == {"c": 1, "y": 1024, "x": 1024}
        assert parse.call_args.kwargs["scale_factors"] is None

    def test_default_3d_chunks_hold_one_channel_and_plane(self, mocker: MockerFixture) -> None:
        data = da.zeros((3, 2, 11, 13), chunks=(3, 2, 11, 13), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("c", "z", "y", "x")), None),
        )
        parse = mocker.patch(
            "spatialdata_io.readers.generic._images.Image3DModel.parse",
            return_value=mocker.sentinel.image,
        )

        result = image("image.tif", "czyx", "global")

        assert result is mocker.sentinel.image
        assert parse.call_args.kwargs["chunks"] == {"c": 1, "z": 1, "y": 1024, "x": 1024}

    @pytest.mark.parametrize(
        ("chunks", "expected"),
        [
            (4, {"c": 4, "z": 4, "y": 4, "x": 4}),
            ((1, 2, 5, 7), {"c": 1, "z": 2, "y": 5, "x": 7}),
            ({"x": 7, "c": 1, "y": 5, "z": 2}, {"c": 1, "z": 2, "y": 5, "x": 7}),
        ],
    )
    def test_accepts_custom_3d_chunks_in_canonical_order(
        self,
        mocker: MockerFixture,
        chunks: Chunks,
        expected: dict[str, int],
    ) -> None:
        data = da.zeros((2, 11, 13, 3), chunks=(2, 11, 13, 3), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("z", "y", "x", "c")), None),
        )
        parse = mocker.patch(
            "spatialdata_io.readers.generic._images.Image3DModel.parse",
            return_value=mocker.sentinel.image,
        )

        result = image("image.tif", "zyxc", "global", chunks=chunks)

        assert result is mocker.sentinel.image
        assert parse.call_args.kwargs["dims"] == ("c", "z", "y", "x")
        assert parse.call_args.kwargs["chunks"] == expected

    def test_forwards_owned_scale_factors_to_the_model(self, mocker: MockerFixture) -> None:
        data = da.zeros((8, 9), chunks=(8, 9), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("y", "x")), None),
        )
        parse = mocker.patch(
            "spatialdata_io.readers.generic._images.Image2DModel.parse",
            return_value=mocker.sentinel.image,
        )
        factors = [2, 4]

        image("image.tif", "yx", "global", scale_factors=factors)
        factors.append(8)

        assert parse.call_args.kwargs["scale_factors"] == [2, 4]

    def test_rejects_invalid_scale_factors_before_reading(self) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            image("missing.tif", "yx", "global", scale_factors=(1,))

    @pytest.mark.parametrize(("scale_factors", "valid"), [((2, 4), True), ((2, 8), False)])
    def test_rejects_pyramid_below_one_pixel_before_model_construction(
        self, mocker: MockerFixture, scale_factors: tuple[int, ...], *, valid: bool
    ) -> None:
        data = da.zeros((5, 8, 9), chunks=(5, 8, 9), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("z", "y", "x")), None),
        )
        parse = mocker.patch(
            "spatialdata_io.readers.generic._images.Image3DModel.parse",
            return_value=mocker.sentinel.image,
        )

        if valid:
            assert image("image.tif", "zyx", "global", scale_factors=scale_factors) is mocker.sentinel.image
            return
        with pytest.raises(ValueError, match=r"exceed the spatial extent of the image \(8 x 9 pixels\)"):
            image("image.tif", "zyx", "global", scale_factors=scale_factors)
        parse.assert_not_called()

    def test_forwards_tiff_selection_once(self, tmp_path: Path, mocker: MockerFixture) -> None:
        path = tmp_path / "image.btf"
        path.touch()
        metadata = mocker.Mock(ome=None)
        data = da.zeros((3, 5, 7), chunks=(1, 5, 7), dtype=np.uint8)
        inspect = mocker.patch(
            "spatialdata_io.readers.generic._images.inspect_tiff",
            return_value=metadata,
        )
        read = mocker.patch(
            "spatialdata_io.readers.generic._images.read_tiff",
            return_value=LazyRaster(data=data, axes=("c", "y", "x")),
        )

        image(path, "cyx", "global", tiff_series=2, tiff_level=1)

        inspect.assert_called_once_with(path.resolve(), context=ReaderErrorContext("generic", "image"))
        read.assert_called_once_with(metadata, series=2, level=1, context=ReaderErrorContext("generic", "image"))

    def test_rejects_missing_and_directory_sources(self, tmp_path: Path) -> None:
        directory = tmp_path / "image.png"
        directory.mkdir()

        with pytest.raises(FileNotFoundError, match=r"reader 'generic', artifact 'image'") as missing:
            image(tmp_path / "missing.tif", "yx", "global")
        with pytest.raises(IsADirectoryError, match="expected an existing regular file; found a directory"):
            image(directory, "yx", "global")

        assert missing.value.filename == str(tmp_path / "missing.tif")

    @pytest.mark.parametrize(("tiff_series", "tiff_level"), [(None, 1), (0, 0)])
    @pytest.mark.parametrize("suffix", ["png", "jpg", "jpeg"])
    def test_rejects_tiff_selection_for_encoded_images(
        self, suffix: str, tiff_series: int | None, tiff_level: int
    ) -> None:
        with pytest.raises(ValueError, match="only select TIFF"):
            image(f"image.{suffix}", "yx", "global", tiff_series=tiff_series, tiff_level=tiff_level)

    @pytest.mark.parametrize(
        "axes",
        ["", "x", "cy", "tcyx", "cyyx", "yxx", "syx", "channel,y,x"],
    )
    def test_rejects_unsupported_or_malformed_axes(self, axes: str) -> None:
        message = f"data_axes must be a permutation of 'yx', 'cyx', 'zyx', or 'czyx'; found {axes!r}."
        with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
            image("image.tif", axes, "global")

    @pytest.mark.parametrize(
        ("axes", "found"),
        [(None, "NoneType"), (3, "int"), (("c", "y", "x"), "tuple"), (["y", "x"], "list"), (b"yx", "bytes")],
    )
    def test_rejects_nonstring_axes(self, axes: object, found: str) -> None:
        with pytest.raises(TypeError, match=f"^data_axes must be a string such as 'cyx'; found {found}.$"):
            image("image.tif", cast("str", axes), "global")

    def test_axes_of_another_rank_are_a_caller_error_with_context(self, tmp_path: Path) -> None:
        path = tmp_path / "image.tif"
        tifffile.imwrite(path, np.zeros((2, 4, 5), dtype=np.uint8), photometric="minisblack", metadata={"axes": "CYX"})

        # Every supported axis set is a subset of its canonical target, so only the rank can disagree.
        with pytest.raises(ValueError, match="source_axes rank 2 does not match array rank 3") as caught:
            image(path, "yx", "global")

        assert type(caught.value) is ValueError
        assert caught.value.__notes__ == [
            f"While handling {ReaderErrorContext('generic', 'image').describe(path=path)}."
        ]

    @pytest.mark.parametrize(
        ("tiff_series", "tiff_level", "error", "match"),
        [
            (-1, 0, ValueError, r"tiff_series must be nonnegative; found -1\."),
            (0, -1, ValueError, r"tiff_level must be nonnegative; found -1\."),
            (True, 0, TypeError, "tiff_series must be an integer or None; found bool"),
            ("0", 0, TypeError, "tiff_series must be an integer or None; found str"),
            (0, 1.0, TypeError, "tiff_level must be an integer; found float"),
        ],
    )
    def test_rejects_invalid_tiff_selection(
        self,
        tiff_series: int,
        tiff_level: int,
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            image(
                "image.tif",
                "yx",
                "global",
                tiff_series=tiff_series,
                tiff_level=tiff_level,
            )

    @pytest.mark.parametrize(
        ("chunks", "error", "match"),
        [
            (False, TypeError, "one positive integer per canonical axis"),
            (0, ValueError, "must be positive"),
            ((1, 0, 7), ValueError, "must be positive"),
            ({"c": True, "y": 5, "x": 7}, TypeError, "one positive integer per canonical axis"),
            ({"c": None, "y": 5, "x": 7}, TypeError, "one positive integer per canonical axis"),
            ({"c": 1, "y": (3, 2), "x": 7}, TypeError, "one positive integer per canonical axis"),
            (((1,), (3, 2), (7,)), TypeError, "one positive integer per canonical axis"),
            ((1, 5), ValueError, "length must match canonical axes"),
            ([1, 5, 7], TypeError, "one positive integer per canonical axis"),
        ],
        ids=[
            "bool",
            "zero",
            "zero-size",
            "bool-size",
            "none-size",
            "irregular-mapping",
            "irregular-tuple",
            "two-sizes",
            "list",
        ],
    )
    def test_rejects_invalid_chunk_values(
        self,
        chunks: object,
        error: type[Exception],
        match: str,
    ) -> None:
        with pytest.raises(error, match=match):
            image("image.tif", "cyx", "global", chunks=cast("Chunks", chunks))

    @pytest.mark.parametrize(
        "chunks",
        [
            {"c": 1, "y": 5},
            {"c": 1, "y": 5, "x": 7, "z": 2},
        ],
    )
    def test_rejects_noncanonical_chunk_mapping(
        self,
        mocker: MockerFixture,
        chunks: dict[str, int],
    ) -> None:
        data = da.zeros((3, 5, 7), chunks=(3, 5, 7), dtype=np.uint8)
        mocker.patch(
            "spatialdata_io.readers.generic._images._read_raster",
            return_value=(LazyRaster(data=data, axes=("c", "y", "x")), None),
        )

        with pytest.raises(ValueError, match="chunks keys must match canonical axes"):
            image("image.tif", "cyx", "global", chunks=chunks)

    def test_rejects_unsupported_image_suffix(self) -> None:
        with pytest.raises(ValueError, match="Unsupported image suffix"):
            image("image.bmp", "yx", "global")


class TestImageArchitecture:
    """Prevent the generic package from regaining pixel-reader ownership."""

    def test_contains_no_legacy_or_eager_raster_calls(self) -> None:
        package = Path("src/spatialdata_io/readers/generic")
        forbidden_imports = {"dask_image", "imageio", "PIL", "rasterio", "rioxarray", "tifffile"}
        forbidden_calls = {"asarray", "compute", "imread", "load", "memmap", "persist"}

        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text())
            imported: list[str] = []
            for node in ast.walk(tree):
                imported.extend(self._imported_modules(node))

            imports = {module.split(".")[0] for module in imported}
            calls = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }

            assert imports.isdisjoint(forbidden_imports), path
            assert calls.isdisjoint(forbidden_calls), path

    @staticmethod
    def _imported_modules(node: ast.AST) -> tuple[str, ...]:
        if isinstance(node, ast.Import):
            return tuple(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            return (node.module,)
        return ()
