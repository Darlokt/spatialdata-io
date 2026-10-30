"""ISS raster options, scientific axes, lazy graphs, and pyramid safety."""

from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest
import tifffile
from dask.callbacks import Callback
from spatialdata.transformations import Identity, get_transformation
from xarray import DataTree

import spatialdata_io.readers._utils.image.tiff._dask as tiff_dask
from spatialdata_io.readers._utils.errors import ReaderFormatError
from spatialdata_io.readers._utils.image import make_raster_options
from spatialdata_io.readers._utils.paths import ArtifactFile
from spatialdata_io.readers.iss._images import _read_raster, _read_rasters, _validate_data_axes
from spatialdata_io.readers.iss._layout import _discover_layout
from tests.integration.readers.iss.conftest import write_iss_dataset

if TYPE_CHECKING:
    from pathlib import Path

    from pytest_mock import MockerFixture


class TestValidateDataAxes:
    """Validate and normalize the explicit raw-image axes without opening artifacts."""

    @pytest.mark.parametrize(("axes", "expected"), [(None, None), ("CYX", "cyx"), ("YXS", "yxs")])
    def test_valid_axes(self, axes: str | None, expected: str | None) -> None:
        assert _validate_data_axes(axes) == expected

    @pytest.mark.parametrize("axes", ["", "CCYX", "QX", "C1YX", "C YX", "CÝX"])
    def test_invalid_axis_names(self, axes: str) -> None:
        with pytest.raises(ValueError, match="data_axes"):
            _validate_data_axes(axes)

    def test_invalid_axes_type(self) -> None:
        with pytest.raises(TypeError, match="data_axes"):
            _validate_data_axes(["c", "y", "x"])  # type: ignore[bad-argument-type, arg-type]  # Exercise an untyped caller boundary.


class TestReadRaster:
    """Inspect source series and explicit axis semantics without decoding pixels."""

    def test_malformed_tiff_has_context(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.tiff"
        path.write_bytes(b"not TIFF")

        with pytest.raises(ValueError, match="TIFF") as caught:
            _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"))

        assert any("iss" in note and str(path) in note for note in caught.value.__notes__)

    def test_multiple_series_are_ambiguous(self, tmp_path: Path) -> None:
        path = tmp_path / "multiple.tiff"
        with tifffile.TiffWriter(path) as output:
            for value in (0, 1):
                output.write(
                    np.full((2, 8, 12), value, dtype=np.uint16), photometric="minisblack", metadata={"axes": "CYX"}
                )

        with pytest.raises(ReaderFormatError, match="2 series"):
            _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"))

    def test_unknown_stack_axis_requires_explicit_override(self, tmp_path: Path) -> None:
        path = tmp_path / "stack.tiff"
        source = np.arange(2 * 8 * 12, dtype=np.uint16).reshape(2, 8, 12)
        tifffile.imwrite(path, source, photometric="minisblack", metadata={"axes": "QYX"})

        with pytest.raises(ReaderFormatError, match="Axis 'q'"):
            _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"))

        image, _ = _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"), data_axes="CYX")
        np.testing.assert_array_equal(image.compute(), source)

    def test_samples_need_an_explicit_channel_interpretation(self, tmp_path: Path) -> None:
        path = tmp_path / "samples.tiff"
        source = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
        tifffile.imwrite(path, source, photometric="rgb")

        with pytest.raises(ReaderFormatError, match="Axis 's'"):
            _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"))

        image, _ = _read_raster(ArtifactFile(path), name="raw image", target_axes=("c", "y", "x"), data_axes="YXC")
        np.testing.assert_array_equal(image.compute(), source.transpose(2, 0, 1))


class TestReadRasters:
    """Validate alignment, source channels, lazy selections and exact label pyramids."""

    @pytest.mark.parametrize("invalid", ["shape", "float", "factor", "uint64", "int64"])
    def test_unsupported_label_contract_rejects_before_pixels(
        self, tmp_path: Path, mocker: MockerFixture, invalid: str
    ) -> None:
        write_iss_dataset(tmp_path)
        dtype = (
            np.float32
            if invalid == "float"
            else np.uint64
            if invalid == "uint64"
            else np.int64
            if invalid == "int64"
            else np.uint16
        )
        shape = (4, 5) if invalid == "shape" else (32, 40)
        tifffile.imwrite(tmp_path / "labels.tiff", np.zeros(shape, dtype=dtype), photometric="minisblack")

        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        factors = (64,) if invalid == "factor" else (2,) if invalid in {"uint64", "int64"} else ()

        with pytest.raises(ValueError, match=r"extent|integer label|32-bit"):
            _read_rasters(layout, data_axes=None, raster=make_raster_options(factors))
        pixels.assert_not_called()

    @pytest.mark.parametrize("names", [["DAPI", "CD3"], ["DAPI", "DAPI"], None])
    def test_ome_names_and_identifiers(self, tmp_path: Path, names: list[str] | None) -> None:
        dataset = write_iss_dataset(tmp_path)
        metadata = {"axes": "CYX"} if names is None else {"axes": "CYX", "Channel": {"Name": names}}
        tifffile.imwrite(tmp_path / "raw.tiff", dataset.raw, ome=True, photometric="minisblack", metadata=metadata)

        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        result = _read_rasters(layout, data_axes=None, raster=make_raster_options())
        expected = names if names is not None and len(set(names)) == 2 else ["Channel:0:0", "Channel:0:1"]
        assert result.image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.image.data.compute(), dataset.raw)

    @pytest.mark.parametrize("ome", [False, True])
    @pytest.mark.parametrize("samples", [3, 4])
    def test_explicit_sample_expansion_preserves_values_and_identities(
        self, tmp_path: Path, *, ome: bool, samples: int
    ) -> None:
        write_iss_dataset(tmp_path)
        source = np.arange(32 * 40 * samples, dtype=np.uint8).reshape(32, 40, samples)
        tifffile.imwrite(tmp_path / "raw.tiff", source, ome=ome, photometric="rgb", metadata={"axes": "YXS"})
        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        with pytest.raises(ReaderFormatError, match="Axis 's'"):
            _read_rasters(layout, data_axes=None, raster=make_raster_options())
        result = _read_rasters(layout, data_axes="yxc", raster=make_raster_options())
        expected = [f"Channel:0:0:sample:{index}" for index in range(samples)] if ome else list(range(samples))
        assert result.image.coords["c"].to_numpy().tolist() == expected
        np.testing.assert_array_equal(result.image.data.compute(), source.transpose(2, 0, 1))

    @pytest.mark.parametrize("compression", [None, "deflate"])
    def test_lazy_native_tiles_are_serializable_and_bounded(
        self, tmp_path: Path, mocker: MockerFixture, compression: str | None
    ) -> None:
        write_iss_dataset(tmp_path)
        source = np.arange(2 * 48 * 64, dtype=np.uint16).reshape(2, 48, 64)
        labels = np.zeros((48, 64), dtype=np.uint16)
        labels[1:4, 2:6] = 7
        tifffile.imwrite(
            tmp_path / "raw.tiff",
            source,
            photometric="minisblack",
            metadata={"axes": "CYX"},
            tile=(16, 16),
            compression=compression,
        )
        tifffile.imwrite(
            tmp_path / "labels.tiff", labels, photometric="minisblack", tile=(16, 16), compression=compression
        )

        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        # Chunks equal to the 16-pixel tiles keep one task per stored tile.
        raster = make_raster_options(chunks=(1, 16, 16))
        pixels = mocker.spy(tiff_dask, "_read_tiff_selection")
        tasks = mocker.Mock()

        with Callback(posttask=tasks):
            result = _read_rasters(layout, data_axes=None, raster=raster)
        tasks.assert_not_called()
        pixels.assert_not_called()

        image = pickle.loads(pickle.dumps(result.image.data))  # noqa: S301, RUF100 - Test-owned graph.

        assert image.chunks == ((1, 1), (16, 16, 16), (16, 16, 16, 16))
        np.testing.assert_array_equal(image[:, :4, :6].compute(), source[:, :4, :6])
        assert pixels.call_count == 2
        np.testing.assert_array_equal(result.labels.data[:4, :6].compute(), labels[:4, :6])
        assert pixels.call_count == 3

    @pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.int32])
    @pytest.mark.parametrize("factors", [(2, 2, 2), (3, 2)])
    def test_every_label_pyramid_level_preserves_dtype_limit_ids(
        self, tmp_path: Path, dtype: type[np.uint16 | np.uint32 | np.int32], factors: tuple[int, ...]
    ) -> None:
        write_iss_dataset(tmp_path)
        identifiers = np.asarray([1, 3, np.iinfo(dtype).max - 1, np.iinfo(dtype).max], dtype=dtype)
        source = np.repeat(np.repeat(identifiers.reshape(2, 2), 32, axis=0), 32, axis=1)
        tifffile.imwrite(
            tmp_path / "raw.tiff",
            np.zeros((2, 64, 64), dtype=np.uint16),
            photometric="minisblack",
            metadata={"axes": "CYX"},
            tile=(16, 16),
        )
        tifffile.imwrite(tmp_path / "labels.tiff", source, photometric="minisblack", tile=(16, 16))

        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        result = _read_rasters(layout, data_axes=None, raster=make_raster_options(factors))

        assert isinstance(result.labels, DataTree)
        for level in result.labels.children.values():
            values = level["image"].data.compute()
            assert values.dtype == np.dtype(dtype)
            np.testing.assert_array_equal(np.unique(values), identifiers)
            np.testing.assert_array_equal(values[np.ix_([0, -1], [0, -1])], identifiers.reshape(2, 2))


class TestPhysicalTransform:
    """Require the original semantic grid and reconcile even partial declarations."""

    def test_spatial_axis_reinterpretation_keeps_pixels(self, tmp_path: Path) -> None:
        fixture = write_iss_dataset(tmp_path, height=40)
        tifffile.imwrite(
            tmp_path / "raw.tiff",
            fixture.raw,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": 0.3, "PhysicalSizeY": 0.7},
        )
        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        result = _read_rasters(layout, data_axes="cxy", raster=make_raster_options())
        assert get_transformation(result.image, to_coordinate_system="global") == Identity()
        assert get_transformation(result.labels, to_coordinate_system="global") == Identity()
        np.testing.assert_array_equal(result.image.data.compute(), fixture.raw.transpose(0, 2, 1))

    def test_partial_raw_and_labels_conflict(self, tmp_path: Path) -> None:
        fixture = write_iss_dataset(tmp_path)
        tifffile.imwrite(
            tmp_path / "raw.tiff",
            fixture.raw,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "CYX", "PhysicalSizeX": 0.3},
        )
        tifffile.imwrite(
            tmp_path / "labels.tiff",
            fixture.labels,
            ome=True,
            photometric="minisblack",
            metadata={"axes": "YX", "PhysicalSizeX": 0.4},
        )
        layout = _discover_layout(
            tmp_path, raw_image_file="raw.tiff", labels_file="labels.tiff", table_file="cells.h5ad"
        )
        with pytest.raises(ReaderFormatError, match="calibration"):
            _read_rasters(layout, data_axes=None, raster=make_raster_options())
